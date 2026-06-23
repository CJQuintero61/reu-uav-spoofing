"""
random_forest.py
06/22/2026

This file implements the random forest classifier
"""
from drone_basics.abstracts import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold

class RandomForestModel(AbstractModel):
    SEED = 0
    SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']


    def __init__(self):
        super().__init__()

        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(n_estimators = 100, random_state = self.SEED, class_weight = 'balanced')

        # Stratified K-Fold cross-validator
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.SEED)
    

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

        print("\nRandom Forest Model Evaluation:")
        print(f"Random Forest Accuracy:  {self.accuracy}")
        print(f"Random Forest Precision: {self.precision}")
        print(f"Random Forest Recall:    {self.recall}")
        print(f"Random Forest F1:        {self.f1}")
        print(f"Random Forest Confusion Matrix:\n{self.confussion_max}")

        # after performing normal evaluation, perform cross validation
        self.scores = self.cross_validate(data)
        print("\nCross Validation Scores:")
        for metric in self.SCORING:
            print(f"{metric} mean: {self.scores[f'{metric}_mean']} +/- {self.scores[f'{metric}_std']}")

        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max
    

    def cross_validate(self, data) -> dict:
        """
        Perform cross validation on the model using the provided data.

        Args:
            data: The data to use for cross validation.

        Returns:
            A dictionary containing the cross validation results.
        """
        scores = {}
        results = cross_validate(
            self.model,
            data.X_Train,
            data.Y_Train,
            cv=self.skf,
            scoring=self.SCORING
        )

        for metric in self.SCORING:
            scores[f'{metric}_mean'] = round((100 * results[f'test_{metric}'].mean()), 2) 
            scores[f'{metric}_std'] = round((100 * results[f'test_{metric}'].std()), 2)
        
        return scores