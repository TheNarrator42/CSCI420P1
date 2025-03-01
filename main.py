import pandas as pd
import preprocess
import os.path
import javalang.tokenizer

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

#takes in panda dataframe, list of tokens, int n for size, returns a dictionary
def ngram(dataset,n=3):
    model = {}
    #key:[TotalTokens,["token", frequency, probability],...]
    #goes through the pandas dataframe, tokenizes it and computes the frequency of the next token
    for index, method in dataset.iterrows():
        s=""
        try:
            tokenlist = list(javalang.tokenizer.tokenize(method['Method Code']))
            #creating every possible token from the list
            #there might be an off by 1 error here
            for i in range(len(tokenlist)-n):
                for j in range(i,i+n+1):
                    s += tokenlist[j]
                #there might be an off by 1 error here
                nextToken = tokenlist[i+n+1]
                #new unseen token not in the model
                if s not in model:  
                    model.update({s:[0,[nextToken,1,None]]})
                #otherwise it is in the model, but is a new token
                elif inTuple(model[s], nextToken):
                    model[s].append([nextToken,1,None])
                    model[s][0] += 1
                #otherwise in the model, not a new token
                else:
                    for i in model[s]:
                        if(nextToken == i[0]):
                            i[1]+=1
                            break
        except:
            pass
    for list in model.values():
        #first element is total frequency
        for token in list[1:]:
            token[2]=float(token[1]/list[0])
    return model

#helper function, takes in a nested list returns if a string s is in the first element of the nested list
def inTuple(list, s):
    for i in range(len(list)):
        if(s == list[i][0]):
            return True
    return False

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
    