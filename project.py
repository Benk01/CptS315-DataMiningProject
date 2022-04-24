import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def main():
    train_data = pd.read_csv("./train.csv")
    labels = train_data.columns[2:].values
    print("Read train.csv complete!")
    print("Labels: " + str(labels))
    print(str(train_data.size) + " comments")

    train_data = train_data.loc[np.random.choice(train_data.index, size=2000)]

    train_data["comment_text"] = train_data["comment_text"].apply(cleanComment)
    train_data["comment_text"] = train_data["comment_text"].apply(removeStopWords)
    train_data["comment_text"] = train_data["comment_text"].apply(stemComment)
    print(train_data.head(15))





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