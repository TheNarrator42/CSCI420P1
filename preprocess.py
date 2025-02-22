import argparse
import csv
import javalang
import os
import pandas as pd
import re

from javalang.parse import parse
from javalang.tree import MethodDeclaration
from pydriller import Repository
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

def remove_duplicates(data):
    """Remove duplicate methods based on method content.
      Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Code", keep="first")

def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    data = data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Code"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like setters and getters."""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data["Method Code"].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data


def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from Java methods in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the methods.
        method_column (str): Column name containing the raw Java methods.
        language (str): Programming language for the lexer (e.g., 'java').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Java Method No Comments'.
    """
    # Define a function to remove comments from a single method
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        # Filter out comments using a lambda function
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))


        return clean_code

    # Apply the function to the specified column
    df["Method Code"] = df[method_column].apply(remove_comments)
    return df

def extract_methods_from_java(code):
    """
    Extract methods from Java source code using javalang parser.

    Args:
        code (str): The Java source code.

    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = javalang.parse.parse(code)
        lines = code.splitlines()

        # Traverse the tree to find method declarations
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name

            # Determine the start and end lines of the method
            start_line = node.position.line - 1
            end_line = None

            # Use the body of the method to determine its end position
            if node.body:
                last_statement = node.body[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line

            # Extract method code
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                # If end_line couldn't be determined, extract up to the end of the file
                method_code = "\n".join(lines[start_line:])

            methods.append((method_name, method_code))
    # except javalang.parser.JavaSyntaxError as e:
    #     # TODO: Consider including this but maybe storing the details elsewhere so it doesn't clutter the output
    #     print(f"Error parsing Java code: {repr(e)} caused by the following code: \n{code}")  
    except Exception as e:
        print(f"Error parsing Java code: {str(e)}")
    return methods

def extract_methods_to_dataframe_from_master(repo_list):
    """
    Extract methods from Java files in the master branch of the repos in the list 
    and save the information in a Pandas dataframe for further processing.

    Args:
        repo_list (List: str): List of path to various Git repositories.
    
    Returns:
        pd.DataFrame: Dataframe consisting of extracted Java methods and relevant information
    """
    extracted_methods = []
  
    for repo_path in repo_list:
      print(f"Processing repository: {repo_path}")
      for commit in Repository(repo_path, only_in_branch="master").traverse_commits():
        #   print(f"Processing commit: {commit.hash}")

          #We only look into the modified files. In other words, we are looking into the history of the software system by traversing each commit.
          #Various Generative AI methods for SD have been trained on data collected in this way; for example bug fixing.
          for modified_file in commit.modified_files:
              if modified_file.filename.endswith(".java") and modified_file.source_code:
                  methods = extract_methods_from_java(modified_file.source_code)

                  for method_name, method_code in methods:
                      extracted_methods.append({
                          "Repo Name": repo_path,
                          "Commit Hash": commit.hash,
                          "File Name": modified_file.filename,
                          "Method Name": method_name,
                          "Method Code": method_code,
                          "Commit Link": f"{repo_path}/commit/{commit.hash}"
                      })

    df = pd.DataFrame(extracted_methods)
    return df

def create_dataset(repo_list_file, output_csv):
    repo_list = []
    with open(repo_list_file) as file:
        for line in file:
            repo_list.append(line.rstrip())

    print(f'Extracting Java methods from Git repositories')

    df = extract_methods_to_dataframe_from_master(repo_list)

    print(f'Raw dataframe consisting of {len(df)} Java methods')

    df = remove_duplicates(df)

    df = filter_ascii_methods(df)

    df = remove_outliers(df)

    df = remove_boilerplate_methods(df)

    df = remove_comments_from_dataframe(df, "Method Code", "Java")

    print(f'Final preprocessed dataframe consisting of {len(df)} Java methods')
    print(f'Saving dataframe as CSV file in {output_csv}')

    df.to_csv(output_csv)
    return df

def tokenization(dataset, output_vocab_file):
    """Tokenizing the dataset to create a vocabulary of unique tokens"""

    print(f'Constructing vocabulary for the model')
    lexer = JavaLexer()

    tokens = [t[1] for code in dataset["Method Code"] for t in lexer.get_tokens(code)]
    vocab = list(set(tokens))

    print(f'Vocabulary consists of {len(vocab)} unique tokens')
    print(f'Storing vocabulary in {output_vocab_file}')

    with open(output_vocab_file, 'w') as file:
        for token in vocab:
            file.write(f"{token}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--git_repo_file", type = str, default = 'ghs_repos.txt', help = 'File containing list of Git repos to use for N-gram model, each separated by newline')
    parser.add_argument("--output_csv_file", type = str, default = 'data.csv', help = 'CSV file storing preprocessed extracted Java methods')
    parser.add_argument("--output_vocab_file", type = str, default = 'vocab.txt', help = 'Text file for storing all the unique tokens')
    args = parser.parse_args()

    df = create_dataset(args.git_repo_file, args.output_csv_file)
    tokenization(df, args.output_vocab_file)
