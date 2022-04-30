import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from IPython.display import Markdown, display
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import LabelPowerset
import time

def main():
    train_data = pd.read_csv("./train.csv")
    # test_comments= pd.read_csv("./test.csv")
    # test_labels = pd.read_csv("./test_labels.csv")
    # test_data = pd.concat([test_comments, test_labels], axis=1)
    labels = train_data.columns[2:].values
    print("Read train.csv complete!")
    print("Labels: " + str(labels))
    print(str(train_data.size) + " comments")

    train_data = train_data.loc[np.random.choice(train_data.index, size=5000)]
    # test_data = test_data.loc[np.random.choice(test_data.index, size=600)]

    train_data["comment_text"] = train_data["comment_text"].apply(cleanComment)
    train_data["comment_text"] = train_data["comment_text"].apply(removeStopWords)
    train_data["comment_text"] = train_data["comment_text"].apply(stemComment)

    # test_data["comment_text"] = test_data["comment_text"].apply(cleanComment)
    # test_data["comment_text"] = test_data["comment_text"].apply(removeStopWords)
    # test_data["comment_text"] = test_data["comment_text"].apply(stemComment)

    # print(train_data.shape)
    # print(test_data.shape)
    
    train, test = train_test_split(train_data, random_state=42, test_size = 0.30, shuffle=True)
    train_text = train['comment_text'].apply(lambda x: ' '.join(x))
    test_text = test['comment_text'].apply(lambda x: ' '.join(x))

    # print(train_text.shape)
    # print(test_text.shape)

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)
    
    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['id','comment_text'], axis=1)

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['id','comment_text'], axis=1)
    
    # BINARY RELEVANCE
    br_time = time.time()
    # Initialize the Binary Relevance Classifier, using the Gaussian Naive Bayes
    br_classifier = BinaryRelevance(GaussianNB())
    
    # Train the data, w functions from the same library
    br_classifier.fit(x_train, y_train)
    
    # Make predictions on the testing data, using functions from the BR library
    br_predictions = br_classifier.predict(x_test)
    
    print("Binary Relevance Testing Accuracy = ", accuracy_score(y_test, br_predictions))
    print("Computation Time: ", ((time.time() - br_time) * 1000), "ms")
    print("\n")
    
    # LABEL POWERSET
    lp_time = time.time()
    # Initialize the Label Powerset multi-label classifier, found in the LabelPowerset library
    lp_classifier = LabelPowerset(LogisticRegression())
    
    # Train the data, again with functions the same library
    lp_classifier.fit(x_train, y_train)
    
    # make predictions on the testing data, again using functions from the LP library
    lp_prediction = lp_classifier.predict(x_test)
    
    print("Label Powerset Testing Accuracy = ", accuracy_score(y_test, lp_prediction))
    print("Computation Time: ", ((time.time() - lp_time) * 1000), "ms")
    print("\n")

# def tf_idf(text):
    # vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    # vectorizer.fit(text)
    # x = vectorizer.transform(text)
    # return x

# cleans comment by setting it to lower case and removing non-alphabetic/non-numeric characters
def cleanComment(comment):
    newComment = ""
    for c in comment:
        if (c.isalpha() or c == '\'' or c.isnumeric()):
            newComment += c
        else:
            newComment += ' '
    return newComment
    
def removeStopWords(comment):
    stop_words = set(stopwords.words('english'))
    word_tokens = tokenize(comment)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return filtered_sentence

def tokenize(comment):
    return comment.split()

def stemComment(comment):
    newComment = []
    ps = PorterStemmer()
    for w in comment:
        newComment.append(ps.stem(w))
    return newComment

main()