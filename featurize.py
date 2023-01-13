from sklearn.feature_extraction.text import TfidfVectorizer


import fluidml
from fluidml.common import Task
from fluidml.swarm import Swarm
from fluidml.flow import Flow, GridTaskSpec, TaskSpec


class FeaturizeText(Task):
    def __init__(self):
        super().__init__()
        self.tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

        self.publishes = ["featurized_text"]

    def run(self, preprocessed_text):
        x_train, x_test, y_train, y_test = preprocessed_text
        self.tfidfvectorizer = self.tfidfvectorizer.fit(x_train+x_test)
        x_train_vec = self.tfidfvectorizer.transform(x_train)
        x_test_vec = self.tfidfvectorizer.transform(x_test)
        print(f"2. Text is featurized.")
        self.save(obj=(x_train_vec, x_test_vec, y_train, y_test), name="featurized_text")
