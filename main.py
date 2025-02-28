import pandas as pd
import preprocess
import os.path
import testing, training, validation

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
    