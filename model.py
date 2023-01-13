import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

import fluidml
from fluidml.common import Task
from fluidml.swarm import Swarm


class TrainModel(Task):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression()

    def train_model(self, text, label):
        self.model = self.model.fit(text, label)

    def evaluate(self, label, predicted_label):
        accuracy = round(accuracy_score(label, predicted_label), 2)
        precision = round(precision_score(label, predicted_label, average="macro"), 2)
        recall = round(recall_score(label, predicted_label, average="macro"), 2)
        f1score = round(f1_score(label, predicted_label, average="macro"), 2)
        mcc = round(matthews_corrcoef(label, predicted_label), 2)
        return accuracy, precision, recall, f1score, mcc

    def run(self, featurized_text):
        x_train, x_test, y_train, y_test = featurized_text
        self.train_model(x_train, y_train)
        y_test_pred = self.model.predict(x_test)
        accuracy, precision, recall, f1score, mcc = self.evaluate(y_test, y_test_pred)
        print("3. Model is trained.")
        print(
            f"\nModel has the following test performance: \n Accuracy:  {accuracy} \n Precision: {precision} \
        \n Recall:    {recall} \n F1-Score:  {f1score} \n MCC:       {mcc}"
        )
        self.save(obj=self.model, name="model")
