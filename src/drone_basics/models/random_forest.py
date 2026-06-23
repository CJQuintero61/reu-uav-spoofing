"""
random_forest.py
06/22/2026

This file implements the random forest classifier
"""
import time
import pickle
from drone_basics.abstracts import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_validate, StratifiedKFold

class RandomForestModel(AbstractModel):
    SEED = 0
    ROUND_PRECISION = 4
    SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'matthews_corrcoef']


    def __init__(self):
        super().__init__()

        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(n_estimators = 100, random_state = self.SEED, class_weight = 'balanced')

        # Stratified K-Fold cross-validator
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.SEED)
    

    def train_model(self, data):
        """train the model and calculate its training time"""
        start = time.time()
        self.model_fit = self.model.fit(data.X_Train, data.Y_Train)
        self.train_time = time.time() - start
    

    def predict(self, data):
        """predict the labels for the test data and calculate its prediction time"""
        start = time.time()
        self.model_prediction = self.model_fit.predict(data.X_Test)
        self.predict_time = time.time() - start
        return self.model_prediction
    

    def evaluate(self, data):
        self.accuracy = accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)
        self.mcc = matthews_corrcoef(data.Y_Test, self.model_prediction)

        self.scores = self._cross_validate(data)    # cross validate
        self._set_model_size()                      # set model size
        self.print_model_info()


        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max
    

    def _cross_validate(self, data) -> dict:
        """
        Perform cross validation on the model using the provided data.

        Args:
            data: The data to use for cross validation.

        Returns:
            A dictionary containing the cross validation results.
        """
        cv_scores = {}
        results = cross_validate(
            self.model,
            data.X_Train,
            data.Y_Train,
            cv=self.skf,
            scoring=self.SCORING
        )

        for metric in self.SCORING:
            cv_scores[f'{metric}_mean'] = results[f'test_{metric}'].mean()
            cv_scores[f'{metric}_std'] = results[f'test_{metric}'].std()
        
        self.cv_scores = cv_scores
        
        return cv_scores


    def _set_model_size(self):
        """calculate the size of the model in KB"""
        model_bytes = len(pickle.dumps(self.model_fit))
        self.model_size_kb = model_bytes / 1024
    

    def print_model_info(self):
        """print the model information"""

        print("\nRandom Forest Model Information:")
        print(f"Model Size:         {round(self.model_size_kb, self.ROUND_PRECISION)} KB")
        print(f"Training Time:      {round(self.train_time, self.ROUND_PRECISION)} seconds")
        print(f"Prediction Time:    {round(self.predict_time, self.ROUND_PRECISION)} seconds")

        print("\nRandom Forest Model Evaluation:")
        print(f"Random Forest Accuracy:  {round(self.accuracy, self.ROUND_PRECISION)}")
        print(f"Random Forest Precision: {round(self.precision, self.ROUND_PRECISION)}")
        print(f"Random Forest Recall:    {round(self.recall, self.ROUND_PRECISION)}")
        print(f"Random Forest F1:        {round(self.f1, self.ROUND_PRECISION)}")
        print(f"Random Forest MCC:       {round(self.mcc, self.ROUND_PRECISION)}")
        print(f"Random Forest Confusion Matrix:\n{self.confussion_max}")

        print("\nCross Validation Scores:")
        for metric in self.SCORING:
            print(f"{metric} Mean: {round(self.cv_scores[f'{metric}_mean'], self.ROUND_PRECISION)}")
            print(f"{metric} Std:  {round(self.cv_scores[f'{metric}_std'], self.ROUND_PRECISION)}\n")
