"""
svc.py
06/22/2026

This file implements the support vector classifier
"""
import time
import pickle
from drone_basics.abstracts import AbstractModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef


class SVCModel(AbstractModel):
    """
    SVC model for classifying spoofing and benign flight data.

    NOTE: this model uses a pipeline to scale the data before fitting the SVC model.
    DO NOT call ReadFlightData.scale_data() before fitting this model, as the pipeline will
    handle scaling the data for each fold in the cross validation, and for the final fit and predict.
    """
    SEED = 0
    ROUND_PRECISION = 4
    SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'matthews_corrcoef']


    def __init__(self):
        super().__init__()
        
        # pipeline for scaling and SVC model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel = 'rbf', class_weight = 'balanced', random_state=self.SEED))
        ])

        # Stratified K-Fold cross-validator
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.SEED)
    

    def train_model(self, data):
        """train the model and calculate its training time"""
        start = time.time()
        self.model_fit = self.pipeline.fit(data.X_Train, data.Y_Train)
        self.train_time = time.time() - start
    

    def predict(self, data):
        """predict the labels for the test data and calculate its prediction time"""
        start = time.time()
        self.model_prediction = self.model_fit.predict(data.X_Test)
        self.predict_time = time.time() - start
        return self.model_prediction


    def evaluate(self, data):
        self.accuracy= accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)
        self.mcc = matthews_corrcoef(data.Y_Test, self.model_prediction)

        self.cv_scores = self._cross_validate(data)  # cross validate
        self._set_model_size()                       # calculate the model size in KB
        self.print_model_info()
        
        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max

    def _cross_validate(self, data) -> dict:
        """
        perform cross validation using the pipeline and the stratified k-fold cross-validator

        returns: dict - a dictionary containing the mean and standard deviation of each metric in SCORING
            ex) 
            {
                'accuracy_mean': 1,
                'accuracy_std': 0,
                'precision_mean': 1,
                'precision_std': 0,
                'recall_mean': 1,
                'recall_std': 0,
                'f1_mean': 1,
                'f1_std': 0,
                'matthews_corrcoef_mean': 1,
                'matthews_corrcoef_std': 0
            }
        """
        scores = {}
        results = cross_validate(
            self.pipeline,
            data.X_Train,
            data.Y_Train,
            cv=self.skf,
            scoring=self.SCORING
        )

        for metric in self.SCORING:
            scores[f'{metric}_mean'] = results[f'test_{metric}'].mean()
            scores[f'{metric}_std'] = results[f'test_{metric}'].std()

        return scores


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

        print("\nCross Validation Scores:")
        for metric in self.SCORING:
            print(f"{metric} mean: {round(self.cv_scores[f'{metric}_mean'], self.ROUND_PRECISION)}")
            print(f"{metric} std:  {round(self.cv_scores[f'{metric}_std'], self.ROUND_PRECISION)}\n")