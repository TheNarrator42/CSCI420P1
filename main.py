import pandas as pd
import preprocess
import os.path
import javalang.tokenizer
import math
import pickle
import bz2

#Setting up file paths to the repos we are extracting and where they are stored
repos = "ghs_repos.txt"
datacsv="data.csv"
testingcsv="testingData.csv"
trainingcsv="trainingData.csv"
validationcsv="validationData.csv"
tokenlist="tokens.txt"
unknownToken = "<UNK>"

def saveCompressed_pickle_bz2(data, filename):
    with bz2.BZ2File(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def loadCompressed_pickle_bz2(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        return pickle.load(f)

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
            for i in range(len(tokenlist)-n):
                s=""
                for j in range(i, i+n):
                    s += tokenlist[j] + " "
                s = s.strip()
                nextToken = tokenlist[i+n]
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

def compute_perplexity(model):
    """Computes the perplexity of an n-gram model based on the probabilities computed during training"""
    probs = [token[2] for val in model.values() for token in val[1:]]
    
    N = len(probs)
    
    return 2 ** ((-1/N) * sum([math.log(prob) for prob in probs]))

#Pass in a model, context window, and a set to see how well the model does
def modelGeneration(model, n, dataset):
    for index, method in dataset.iterrows():
        print("New loop")
        seen = set()
        tokenlist = []
        generatedCode = []    
        tokens = list(javalang.tokenizer.tokenize(method['Method Code']))
        tokenlist.extend([token.value for token in tokens])
        try:
            generatedCode = tokenlist[0:n]
        except:
            print("Not enough tokens")
        counter = 0
        while(True):
            try:
                #TODO fix the ngram mmodel generation so there is no trailing space
                key = " ".join(generatedCode[counter:counter+n]+ [""])
                generatedCode.append(model[key][1][0])
                if key in seen:
                    print("Looping")
                    break
                seen.add(key)
                counter +=1
            except Exception as e:
                print(f"Error{type(e)}:  {e}")
                if type(e) == KeyError:
                    generatedCode.append(unknownToken)
                print(generatedCode)
                break
    return 1
    #May change this return statement
    #return (compute_perplexity(generatedCode))
        
        
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

    else:
        print("Data found")
        testingData = pd.read_csv(testingcsv)
        trainingData = pd.read_csv(trainingcsv)
        validationData = pd.read_csv(validationcsv)

    print("Data Seperation:")
    print(testingData)
    print(trainingData)
    print(validationData)

    #Magic unused values
    bestPerp = -1
    bestN = -1
    #Train and obtain a bunch of different models
    for i in range(1,11):
        if(not os.path.isfile(f"{i}-gramData.pkl.bz2")):
            model = ngram(trainingData, i)
            saveCompressed_pickle_bz2(model, f"{i}-gramData.pkl.bz2")
        else:
            model = loadCompressed_pickle_bz2(f"{i}-gramData.pkl.bz2")
        perplexity = compute_perplexity(model)
        print(f"n={i}")
        print(f"Perplexity of training set: {perplexity}")
        if(perplexity < bestPerp or bestPerp == -1):
            bestPerp = perplexity
            bestN = i
    
    print(f"Best Perplexity: {bestPerp}")
    print(f"Best N-gram model is when n = {bestN}")
    
    for i in range(1,11):
        print(f"Model: {i}-gramData.pkl.bz2")
        model = loadCompressed_pickle_bz2(f"{i}-gramData.pkl.bz2")
        modelGeneration(model, i, testingData)
    