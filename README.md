# GenAI for Software Development (N-gram)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)
  * [2.4 Extraction & Preprocessing](#24-extraction--preprocessing)
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

## **2.3 Run N-gram**
The ```main.py``` script accepts a corpus of Java methods to train and N-gram models with varying N values from N =  2 to N = 10 and determines the best-performing model based on perplexity score. The perplexity is computed on a reserved validation set. Next, it carries out code completion on a set of 100 reserved test methods and stores the results in a JSON file. It also computes the perplexity on the test set. All trained N-gram models are saved as pickle files.

Run the script using the following command:

```(venv) ~/CSCI420P1 $ python main.py```

## **2.4 Extraction & Preprocessing**
In addition to creating a script for training, evaluation, and testing N-grams, we also wrote a script for creating a cleaned dataset of Java methods called ```preprocess.py.``` This script is called by ```main.py``` to clean the input training dataset. However, this script can also standalone and be used to create datasets from GitHub repositories. In fact, this script was used to generate our the dataset we used for training, testing, and evaluation.

Given a text file containing a list of GitHub repositories in the directory (by default, the script assumes it's called ```ghs_repos.txt```), it pulls the source code from the repositories and extracts their Java methods. Next, it performs preprocessing on the data, including removing duplicates methods, methods with non-ASCII characters, outlier methods (determined based on method length), boilerplate methods (e.g., getter and setter methods), and comments. The final dataset is a csv file containing the Java methods.

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