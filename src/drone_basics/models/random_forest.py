"""
random_forest.py
06/22/2026

This file implements the random forest classifier
"""
import time
import pickle
import pandas as pd
from abstracts import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

class RandomForestModel(AbstractModel):
    SEED = 0
    ROUND_PRECISION = 4


    def __init__(self, config):
        super().__init__()
        self.n_est = config["n_estimators"]
        self.min_samples_leaf = config["min_samples_leaf"]
        self.max_dep = config["max_depth"] 

        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(
            n_estimators = self.n_est,
            min_samples_leaf = self.min_samples_leaf,
            max_depth=self.max_dep,
            random_state = self.SEED,
        )

        print(f'DEBUG: Random Forest model parameters: {self.model.get_params()}')

    def train_model(self, data):
        """train the model and calculate its training time"""

        print(f'\n-----Begin RF Model Training at {time.strftime("%H:%M:%S", time.localtime())}-----')
        start = time.time()
        self.model_fit = self.model.fit(data.X_Train, data.Y_Train)
        self.training_time = time.time() - start
    

    def predict(self, data):
        """predict the labels for the test data and calculate its prediction time"""

        print(f'\n-----Begin RF Model Prediction at {time.strftime("%H:%M:%S", time.localtime())}-----')
        start = time.time()
        self.model_prediction = self.model_fit.predict(data.X_Test)
        self.testing_time = time.time() - start
        return self.model_prediction
    

    def evaluate(self, data):
        self.accuracy = accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)
        self.mcc = matthews_corrcoef(data.Y_Test, self.model_prediction)

        self._set_model_size()
        self.print_model_info()


        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max


    def _set_model_size(self):
        """calculate the size of the model in KB"""
        model_bytes = len(pickle.dumps(self.model_fit))
        self.model_size = model_bytes / 1024
    

    def print_model_info(self):
        """print the model information"""

        print("\nRandom Forest Model Information:")
        print(f"Model Size:         {round(self.model_size, self.ROUND_PRECISION)} KB")
        print(f"Training Time:      {round(self.training_time, self.ROUND_PRECISION)} seconds")
        print(f"Prediction Time:    {round(self.testing_time, self.ROUND_PRECISION)} seconds")

        print("\nRandom Forest Model Evaluation:")
        print(f"Random Forest Accuracy:  {round(self.accuracy, self.ROUND_PRECISION)}")
        print(f"Random Forest Precision: {round(self.precision, self.ROUND_PRECISION)}")
        print(f"Random Forest Recall:    {round(self.recall, self.ROUND_PRECISION)}")
        print(f"Random Forest F1:        {round(self.f1, self.ROUND_PRECISION)}")
        print(f"Random Forest MCC:       {round(self.mcc, self.ROUND_PRECISION)}")
        print(f"Random Forest Confusion Matrix:\n{self.confussion_max}")

