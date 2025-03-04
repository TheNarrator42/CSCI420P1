# GenAI for Software Development (N-gram)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Training, Evaluation, and Testing N-grams](#23-training-evaluation-and-testing-n-grams)
  * [2.4 Dataset Preparation](#24-dataset-preparation)
* [3 Report](#3-report)  

---

# **1. Introduction** 
This project implements an N-gram model for Java code completion. The N-gram predicts the next token based on the previous N tokens in a sequence. During training, the model computes the probabilities of various N+1 sequences in the training data and uses these probabilities to deduce the most likely token to follow a series of tokens. 

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/TheNarrator42/CSCI420P1.git
```

(2) Navigate into the repository:
```
~ $ cd CSCI420P1
~/CSCI420P1 $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:
```
~/CSCI420P1 $ python -m venv ./venv/
~/CSCI420P1 $ source venv/bin/activate
(venv) ~/CSCI420P1 $ 
```

For Windows:
```
~/CSCI420P1 $ python -m venv ./venv/
~/CSCI420P1 $ .\venv\Scripts\activate
```

To deactivate the virtual environment, use the command:
```
(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:
```shell
(venv) ~/CSCI420P1 $ pip install -r requirements.txt
```

## **2.3 Training, Evaluation, and Testing N-grams**
The ```main.py``` script takes a corpus of Java methods to train and N-gram models with varying N values from N = 1 to N = 10. Then, it evaluates the models by computing their perplexities on a set aside validation set and selects the best-performing model. The script also uses the best-performing mdoel to carry out code completion code completion on a set of 100 test methods and stores the results in a JSON file called ```testingResults.json```. Each entry in the JSON is formatted as follows:
```
{
    "[index of test method]":{
        "Generated Code":"('[predictedToken1]', [probability of predictedToken1]) ('[predictedToken2]', [probability of predictedToken2)], ..."
    }
    ...
}
```

Additionally, the script writes the perplexity of the best performing model on the validation and testing data in a file called ```perplexities.txt```.

There are two options to run the script:
1) **Use pre-made dataset:** We prepared our own dataset, which is provided in the repository (```data.csv```). You can train, validate, and test N-gram models using the pre-made dataset with the following command:

```(venv) ~/CSCI420P1 $ python main.py 1```

Note that if you are running the script for the first time, it will automatically create three separate files: ```trainingData.csv```, ```validationData.csv```, and ```testingData.csv.``` These files can be use be used for future runs.

2) **Provide your own training corpus:** You can also provide your own training data in the form of a text file, where each line is a Java method.

```(venv) ~/CSCI420P1 $ python main.py 2 [path\to\training_data].txt```


As with option 1, if there is no presaved ```validationData.csv``` or ```testingData.csv```, the script will generate these from ```data.csv.```

For either option, the script trains N-gram models and saves them as pickle files (e.g., ```3-gramData.pkl.bz2```) which are stored in a directory called ```gramModels```. Note that the script does **not** overwrite these files in each consecutive run. If you wish to re-run the script with different training data (via option 1 or 2), you must either delete the previously created pickle files or rename them. 

In the repository, we provide the results from our runs on our data and the instructor-provided corpus in the ```githubData``` and ```instructorData``` directories, respectively.

## **2.4 Dataset Preparation**
In addition to creating a script for training, evaluation, and testing N-grams, we also wrote ```preprocess.py``` for dataset preparation. This script is called by ```main.py``` to clean the inputted training dataset. However, this script can also stand alone and be used to create datasets from GitHub repositories. In fact, this script was used to generate the dataset (```data.csv```) we used for training, testing, and evaluation.

Given a text file containing a list of GitHub repositories (by default, the script assumes it's called ```ghs_repos.txt```), it pulls the source code from the repositories and extracts their Java methods. Then, it performs preprocessing on the data, including removing duplicates methods, methods with non-ASCII characters, outlier methods (determined based on method length), boilerplate methods (e.g., getter and setter methods), and comments. The final result is a csv file containing the Java methods.

To run the script, use the following command:

```(venv) ~/CSCI420P1 $ python preprocess.py```

The script also accepts various command-line arguments for the names of input and output files.
```
usage: preprocess.py [-h] [--git_repo_file GIT_REPO_FILE] [--output_csv_file OUTPUT_CSV_FILE]
                     [--output_vocab_file OUTPUT_VOCAB_FILE]

options:
  -h, --help            show this help message and exit
  --git_repo_file GIT_REPO_FILE
                        File containing list of Git repos to use for N-gram model, each separated by newline
  --output_csv_file OUTPUT_CSV_FILE
                        CSV file storing preprocessed extracted Java methods
  --output_vocab_file OUTPUT_VOCAB_FILE
                        Text file for storing all the unique tokens
```


## 3. Report

The assignment report is available in the file **GenAI for Software Development - Assignment 1.pdf**.
