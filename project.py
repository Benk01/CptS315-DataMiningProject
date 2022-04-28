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

    train, test = train_test_split(train_data, random_state=42, test_size=0.30, shuffle=True)

    print(train.shape)
    print(test.shape)

    train_text = train['comment_text'].apply(lambda x: ' '.join(x))
    test_text = test['comment_text'].apply(lambda x: ' '.join(x))

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)
    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['id','comment_text'], axis=1)

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['id','comment_text'], axis=1)
    print(x_train)
    #uniqueWords = train_data["comment_text"].explode().unique()
    #for row in train_data["comment_text"]:

    #uniqueWords = set(train_data["comment_text"])
    #for line in train_data["comment_text"]:
    LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])

    for category in labels:
        
        # Training logistic regression model on train data
        LogReg_pipeline.fit(x_train, train[category])
        
        # calculating test accuracy
        prediction = LogReg_pipeline.predict(x_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
        print("\n")
        



def tf_idf(text):
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(text)
    x = vectorizer.transform(text)
    return x

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