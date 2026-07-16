"""
svc.py
06/22/2026

This file implements the support vector classifier
"""
import time
import pickle
import pandas as pd
from abstracts import AbstractModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef


class SVCModel(AbstractModel):
    """
    SVC model for classifying spoofing and benign flight data.
    """
    SEED = 0
    ROUND_PRECISION = 4


    def __init__(self, config):
        super().__init__()
        
        # pipeline for scaling and SVC model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel = 'rbf',
                class_weight = {0: 1, 1: 2},    # optimal weights for overall metric scores
                C = 1.0,                        # default
                gamma = 'scale',                # default
                random_state=self.SEED
            ))
        ])

        print(f"DEBUG: actual SVC params -> {self.pipeline.named_steps['svc'].get_params()}")
    

    def train_model(self, data):
        """train the model and calculate its training time"""

        print(f'\n-----Begin SVC Model Training at {time.strftime("%H:%M:%S", time.localtime())}-----')
        start = time.time()
        self.model_fit = self.pipeline.fit(data.X_Train, data.Y_Train)
        self.train_time = time.time() - start
    

    def predict(self, data):
        """predict the labels for the test data and calculate its prediction time"""

        print(f'\n-----Begin SVC Model Prediction at {time.strftime("%H:%M:%S", time.localtime())}-----')
        start = time.time()
        self.model_prediction = self.model_fit.predict(data.X_Test)
        self.predict_time = time.time() - start
        return self.model_prediction


    def evaluate(self, data):
        self.accuracy= accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction)
        self.recall = recall_score(data.Y_Test, self.model_prediction)
        self.f1 = f1_score(data.Y_Test, self.model_prediction)
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)
        self.mcc = matthews_corrcoef(data.Y_Test, self.model_prediction)

        self._set_model_size()
        self.print_model_info()
        
        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max


    def _set_model_size(self):
        """calculate the size of the model in KB"""
        model_bytes = len(pickle.dumps(self.model_fit))
        self.model_size_kb = model_bytes / 1024
    

    def print_model_info(self):
        """print the model information"""

        print("\nSVC Model Information:")
        print(f"Model Size:         {round(self.model_size_kb, self.ROUND_PRECISION)} KB")
        print(f"Training Time:      {round(self.train_time, self.ROUND_PRECISION)} seconds")
        print(f"Prediction Time:    {round(self.predict_time, self.ROUND_PRECISION)} seconds")

        print("\nSVC Model Evaluation:")
        print(f"SVC Accuracy:  {round(self.accuracy, self.ROUND_PRECISION)}")
        print(f"SVC Precision: {round(self.precision, self.ROUND_PRECISION)}")
        print(f"SVC Recall:    {round(self.recall, self.ROUND_PRECISION)}")
        print(f"SVC F1:        {round(self.f1, self.ROUND_PRECISION)}")
        print(f"SVC MCC:       {round(self.mcc, self.ROUND_PRECISION)}")
        print(f"SVC Confusion Matrix:\n{self.confussion_max}")