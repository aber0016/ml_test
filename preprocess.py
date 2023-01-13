import pandas as pd
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.model_selection import train_test_split

import fluidml
from fluidml.common import Task
from fluidml.swarm import Swarm


class PreProcessData(Task):
    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name
        self.tokenizer = word_tokenize
        self.stop_words = set(stopwords.words("english"))

        self.publishes = ["preprocessed_text"]

    def read_in_data(self):
        df = pd.read_csv("/home/aberger/Desktop/code/ml_test/data/train.csv")
        return df
    
    def split_data(self, text, label):
        x_train, x_test, y_train, y_test = train_test_split(text, label, train_size=0.8)
        return x_train, x_test, y_train, y_test

    def preprocess_text(self, df):
        # save processed sms in
        processed_text = []

        raw_text = df["sms"].to_list()
        label = df["label"].to_list()

        for sms in raw_text:
            tokens = word_tokenize(sms)
            tokens = [
                token.lower()
                for token in tokens
                if token not in self.stop_words
                and len(token) > 1
                and token.isalnum() == True
            ]
            processed_text.append(" ".join(tokens))
        return processed_text, label

    def run(self):
        df = self.read_in_data()
        text, label = self.preprocess_text(df)
        x_train, x_test, y_train, y_test = self.split_data(text, label)
        print(f"1. {self.file_name} read in and processed.")
        self.save(obj=(x_train, x_test, y_train, y_test), name="preprocessed_text")
