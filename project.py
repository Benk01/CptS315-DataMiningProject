import numpy as np
import pandas as pd

def parseTrain():
    train_data = pd.read_csv("./train.csv")
    labels = train_data.columns[2:].values
    print("Read train.csv complete!")
    print("Labels: " + str(labels))
    print(str(train_data.size) + " comments")
            



if __name__ == "__main__":
    parseTrain()