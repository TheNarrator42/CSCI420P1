import pandas as pd
import preprocess
import os.path
import javalang.tokenizer
import math

#Setting up file paths to the repos we are extracting and where they are stored
repos = "ghs_repos.txt"
datacsv="data.csv"
testingcsv="testingData.csv"
trainingcsv="trainingData.csv"
validationcsv="validationData.csv"
tokenlist="tokens.txt"

def dataSplit(fullData):
    #Splits data
    testingData = fullData.sample(100)
    fullData = fullData.drop(testingData.index)
    trainingData= fullData.sample(frac=0.8)
    fullData = fullData.drop(trainingData.index)
    validationData = fullData.sample(frac=1)
    fullData = fullData.drop(validationData.index)
    print("Data partitioned")
    print(fullData)
    testingData.to_csv(testingcsv, index = False)
    trainingData.to_csv(trainingcsv, index = False)
    validationData.to_csv(validationcsv, index = False)
    return testingData, trainingData, validationData

#takes in panda dataframe, int n for size, returns a dictionary
def ngram(dataset,n=3):
    model = {}
    #key:[TotalTokens,["token", frequency, probability],...]
    #goes through the pandas dataframe, tokenizes it and computes the frequency of the next token
    for index, method in dataset.iterrows():
        tokenlist = []
        try:
            tokens = list(javalang.tokenizer.tokenize(method['Method Code']))
            tokenlist.extend([token.value for token in tokens])
            #creating every possible token from the list
            #there might be an off by 1 error here
            for i in range(len(tokenlist)-n+1):
                s=""
                for j in range(i, i+n-1):
                    s += tokenlist[j] + " "
                
                nextToken = tokenlist[i+n-1]
                #new unseen token not in the model
                if s not in model:
                    model.update({s:[1,[nextToken,1,None]]})
                #otherwise it is in the model, but is a new token
                elif not inList(model[s][1:], nextToken): 
                    model[s].append([nextToken,1,None])
                    model[s][0] += 1
                #otherwise in the model, not a new token
                else:
                    for i in model[s][1:]:
                        if(nextToken == i[0]):
                            i[1]+=1
                            break
                    model[s][0] += 1
        except Exception as e:
            # print(f'Exception while creating n-gram model - {type(e)}: {e}')
            continue
    for val in model.values():
        #first element is total frequency
        for token in val[1:]:
            if val[0] == 0:
                token[2] = 0
            else:
                token[2]=float(token[1]/val[0])
    return model

#helper function, takes in a nested list returns if a string s is in the first element of the nested list
def inList(nested_list, s):
    for i in range(len(nested_list)):
        if(s == nested_list[i][0]):
            return True
    return False

def compute_perplexity(model, dataset,n=3):
    """Computes the perplexity based on the probabilities computed during training"""
    N = 0
    log_prob_sum = 0

    for index, method in dataset.iterrows(): #Iterating through the methods in the dataset and tokenizing each of them
        tokenlist = []
        try:
            tokens = list(javalang.tokenizer.tokenize(method['Method Code'])) #NOTE: Should we have the tokenized prior? Redundant code here
            tokenlist.extend([token.value for token in tokens]) 
            
            N += len(tokenlist)
            for i in range(len(tokenlist)-n+1):
                # Computing log(p(w_i|context)), where context consists of prev n-1 tokens
                context = "" 
                for j in range(i, i+n-1):
                    context += tokenlist[j] + " "

                log_prob = 0
                if context in model: 
                    for next_token in model[context][1:]: #going through all the tokens that the model determined could come after the ngram
                        if tokenlist[i+n-1] == next_token[0]:
                            log_prob = math.log(next_token[2])
                            break
                if log_prob == 0: # Either the context doesn't exist in the model or the token has never been seen before
                    log_prob = 1e-6 #Very small value since probability can't be 0
                log_prob_sum += log_prob                 
                
        except Exception as e:
            print(f'Exception while computing perplexity - {type(e)}: {e}')
            break#continue
    
    return math.exp((-1/N) * log_prob_sum)

if __name__ == '__main__':
    #creates the dataset if it does not exist
    if(not os.path.isfile(datacsv)):
        print("extracting data")
        fullData = preprocess.create_dataset(repos, datacsv)
    else:
        print("data found")
        fullData = pd.read_csv(datacsv)
    #Dropping unneeded columns
    fullData.drop(axis=1, labels=["Commit Hash", "Commit Link","Method Name","File Name", "Repo Name"], inplace=True)
    fullData.drop(fullData.columns[0],axis=1,inplace = True)

    #Tokenization
    if(not os.path.isfile(tokenlist)):
        tokens = preprocess.tokenization(fullData.values.tolist(), tokenlist)
    else:
        tokens = open(tokenlist).read().splitlines()

    #Seperating data into three sets
    if(not os.path.isfile(testingcsv) and not os.path.isfile(trainingcsv) and not os.path.isfile(validationcsv)):
        print("Data not found")
        testingData, trainingData, validationData=dataSplit(fullData)
        print(testingData)
        print(trainingData)
        print(validationData)
    else:
        print("Data found")
        testingData = pd.read_csv(testingcsv)
        trainingData = pd.read_csv(trainingcsv)
        validationData = pd.read_csv(validationcsv)
        print(testingData)
        print(trainingData)
        print(validationData)

    model = ngram(trainingData, 3)
    #TODO: perplexity and stuff here
    print(f"Perplexity of training set: {compute_perplexity(model, trainingData)}")
    print(f"Perplexity of validation set: {compute_perplexity(model, validationData)}")
    print(f"Perplexity of testing set: {compute_perplexity(model, testingData)}")
    