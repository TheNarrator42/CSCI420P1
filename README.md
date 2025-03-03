# GenAI for Software Development (N-gram)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)  
* [3 Report](#3-report)  

---

# **1. Introduction** 
This project explores **code completion in Java**, leveraging **N-gram language modeling**. The N-gram model predicts the next token in a sequence by learning the probability distributions of token occurrences in training data. The model selects the most probable token based on learned patterns, making it a fundamental technique in natural language processing and software engineering automation.

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/TheNarrator42/CSCI420P1.git

(2) Navigate into the repository:

~ $ cd CSCI420P1
~/CSCI420P1 $

(3) Set up a virtual environment and activate it:

For macOS/Linux:

~/CSCI420P1 $ python -m venv ./venv/
~/CSCI420P1 $ source venv/bin/activate
(venv) ~/CSCI420P1 $ 


To deactivate the virtual environment, use the command:

(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:

(venv) ~/CSCI420P1 $ pip install -r requirements.txt

## **2.3 Run N-gram**

(1) Run N-gram Demo

The script takes a corpus of Java methods as input and automatically identifies the best-performing model based on a specific N-value. We implemented two ways for the user to input a corpus. One way is by creating a list of GitHub repositories in a text file. The script will automatically extract Java methods from the master branches in the repositories and produce a preprocessed and cleaned corpus. 

(venv) ~/your-project $ python main.py ghs_repos.txt

Otherwise, the user can input a corpus consisting of a list of Java methods.

(venv) ~/your-project $ python main.py corpus.txt

After receiving a corpus, the script trains N-grams with various values of N and identifies the model with the best perplexity on the validation set. Finally, it evaluates the selected model on the test set, which are extracted based on the assignment specifications. 
Since the training corpus differs from both the instructor-provided dataset and our own dataset, we store the results in a file named results_provided_model.json to distinguish them accordingly.


## 3. Report

The assignment report is available in the file Assignment_Report.pdf.