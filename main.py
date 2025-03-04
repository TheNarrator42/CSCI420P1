import pandas as pd
import preprocess
import os
import javalang.tokenizer
import math
import pickle
import bz2
import sys

#Setting up file paths to the repos we are extracting and where they are stored
repos = "ghs_repos.txt"
datacsv="data.csv"
testingcsv="testingData.csv"
trainingcsv="trainingData.csv"
validationcsv="validationData.csv"
tokenlist="tokens.txt"
unknownToken = "<UNK>"
endToken = "<END>"

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
    print("Data Splited")
    return trainingData, testingData, validationData

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
            #Special case the end of the function to insert a special end token
            endSequence = " ".join(tokenlist[-n:]).strip()
            if endSequence not in model:
                model.update({endSequence:[1,[endToken,1,None]]})
            elif not inList(model[endSequence][1:], endToken):
                model[endSequence].append([endToken,1,None])
                model[endSequence][0] += 1
            else:
                    for i in model[endSequence][1:]:
                        if(endToken == i[0]):
                            i[1]+=1
                            break
                    model[endSequence][0] += 1

            #creating every possible token from the list
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
            #print(f'Exception while creating n-gram model - {type(e)}: {e}')
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
            tokens = list(javalang.tokenizer.tokenize(method['Method Code']))
            tokenlist.extend([token.value for token in tokens]) 
            
            N += len(tokenlist)
            for i in range(len(tokenlist)-n):
                # Computing log(p(w_i|context)), where context consists of prev n-1 tokens
                context = "" 
                for j in range(i, i+n):
                    context += tokenlist[j] + " "
                context = context.strip()
                log_prob = math.log(1e-6) # Default to low probability if either the context doesn't exist in model or the token hasn't been seen before
                if context in model: 
                    for next_token in model[context][1:]: #going through all the tokens that the model determined could come after the ngram
                        if tokenlist[i+n] == next_token[0]:
                            log_prob = math.log(next_token[2])
                            break
                log_prob_sum += log_prob                 
                
        except Exception as e:
            print(f'Exception while computing perplexity - {type(e)}: {e}')
            break#continue
    
    return math.exp((-1/N) * log_prob_sum)

#Pass in a model, context window, and a set to see how well the model does
def modelGeneration(model, dataset, n):
    braces = None
    allGenCode = pd.DataFrame(columns=["Generated Code"])
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
        row_generated = []
        
        while True:
            try:
                key = " ".join(generatedCode[counter:counter+n])
                #Next predicted token
                nextPredict = model[key][1][0]
                generatedCode.append(nextPredict)
                row_generated.append((nextPredict,model[key][1][2]))

                if nextPredict == endToken:
                    print("End token detected, ending method")
                    break
                elif nextPredict == "{":
                    if braces == None :
                        braces=1
                    else:
                        braces+=1
                elif nextPredict == "}":
                    if braces == None :
                        print("Braces mismatched, ending generation")
                        break
                    braces -= 1
                    if braces == 0:
                        print("Braces matched, ending")
                        break

                if key in seen:
                    print("Looping")
                    generatedCode = generatedCode[:-n]
                    break
                seen.add(key)
                counter +=1
            except Exception as e:
                print(f"Error{type(e)}:  {e}")
                if type(e) == KeyError:
                    generatedCode.append(unknownToken)
                print(generatedCode)
                break
        allGenCode = pd.concat([allGenCode, pd.DataFrame([{"Generated Code": " ".join(map(str, row_generated))}])], ignore_index=True)


    return allGenCode
        
        
if __name__ == '__main__':
    #Creates the dataset if it does not exist
    if(not os.path.isfile(datacsv)):
        print("Extracting data")
        fullData = preprocess.create_dataset(repos, datacsv)
    else:
        print("Data found")
        fullData = pd.read_csv(datacsv)
    # Dropping unneeded columns
    fullData.drop(axis=1, labels=["Commit Hash", "Commit Link","Method Name","File Name", "Repo Name"], inplace=True)
    fullData.drop(fullData.columns[0],axis=1,inplace = True)

    #Tokenization
    if(not os.path.isfile(tokenlist)):
        tokens = preprocess.tokenization(fullData.values.tolist(), tokenlist)
    else:
        tokens = open(tokenlist).read().splitlines()

    #Seperating data into three sets
    if (not os.path.isfile(testingcsv) and not os.path.isfile(trainingcsv) and not os.path.isfile(validationcsv)):
        print("Data Seperation not found")
        print("Attempting to split data")
        match int(sys.argv[1]):
            #Reading in data from text file
            case 1:
                trainingData, testingData, validationData=dataSplit(fullData)
            #Reading in data for training from command line
            case 2:
                _, testingData, validationData=dataSplit(fullData)
                trainingData = pd.read_table(sys.argv[2], header=None)
                trainingData = trainingData.rename(columns={trainingData.columns[0]:"Method Code"})
                trainingData = preprocess.data_clean(trainingData)

        testingData.to_csv(testingcsv, index = False)
        trainingData.to_csv(trainingcsv, index = False)
        validationData.to_csv(validationcsv, index = False)
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
    
    if not os.path.exists("gramModels"):
        os.makedirs("gramModels")

    #Train and obtain a bunch of different models
    for i in range(1,11):
        if(not os.path.isfile(f"{i}-gramData.pkl.bz2")):
            model = ngram(trainingData, i)
            saveCompressed_pickle_bz2(model, f"gramModels/{i}-gramData.pkl.bz2")
        else:
            model = loadCompressed_pickle_bz2(f"gramModels/{i}-gramData.pkl.bz2")
        perplexity = compute_perplexity(model, validationData, i)
        print(f"n={i}")
        print(f"Perplexity of validation set: {perplexity}")
        if(perplexity < bestPerp or bestPerp == -1):
            bestPerp = perplexity
            bestN = i
    
    print(f"Best Perplexity: {bestPerp}")
    print(f"Best N-gram model is when n = {bestN}")
    
    model = loadCompressed_pickle_bz2(f"gramModels/{bestN}-gramData.pkl.bz2")
    
    validationPerp = compute_perplexity(model, validationData,bestN)
    testingPerp = compute_perplexity(model, testingData, bestN)
    print(f"Perplexity Data for validation data: {validationPerp}")
    print(f"Perplexity Data for testing data: {testingPerp}")

    with open("perplexities.txt", "w") as f:
        f.write(str(bestN)+"-gram model perplexity: "+str(bestPerp)+"\n")
        f.write(f"Perplexity Data for validation data: {validationPerp} \n")
        f.write(f"Perplexity Data for testing data: {testingPerp} \n")
    
    df = modelGeneration(model, testingData, bestN)
    df.to_json("testingResults.json", orient="index", indent=4)