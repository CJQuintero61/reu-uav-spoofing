"""
random_forest.py
06/22/2026

This file implements the random forest classifier
"""
from abstracts import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class RandomForestModel(AbstractModel):
    SEED = 0

    def __init__(self):
        super().__init__()

        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(n_estimators = 100, random_state = self.SEED, class_weight = 'balanced')
    
    def train_model(self, data):
        self.model_fit = self.model.fit(data.X_Train, data.Y_Train)
    
    def predict(self, data):
        self.model_prediction = self.model_fit.predict(data.X_Test)
        return self.model_prediction
    
    def evaluate(self, data):
        self.accuracy = accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)

        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max