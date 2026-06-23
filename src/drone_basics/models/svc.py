"""
svc.py
06/22/2026

This file implements the support vector classifier
"""
from drone_basics.abstracts import AbstractModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class SVCModel(AbstractModel):
    """
    SVC model for classifying spoofing and benign flight data.

    NOTE: this model uses a pipeline to scale the data before fitting the SVC model.
    DO NOT call ReadFlightData.scale_data() before fitting this model, as the pipeline will
    handle scaling the data for each fold in the cross validation, and for the final fit and predict.
    """
    SEED = 0
    SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

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
        # fit the model using the pipeline
        self.model_fit = self.pipeline.fit(data.X_Train, data.Y_Train)
    
    def predict(self, data):
        self.model_prediction = self.model_fit.predict(data.X_Test)
        return self.model_prediction

    def evaluate(self, data):
        self.accuracy= accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)

        print("\nSVC Model Evaluation:")
        print(f"SVC Accuracy:  {self.accuracy}")
        print(f"SVC Precision: {self.precision}")
        print(f"SVC Recall:    {self.recall}")
        print(f"SVC F1:        {self.f1}")
        print(f"SVC Confusion Matrix:\n{self.confussion_max}")

        # after performing normal evaluation, perform cross validation
        self.scores = self.cross_validate(data)
        print("\nCross Validation Scores:")
        for metric in self.SCORING:
            print(f"{metric} mean: {self.scores[f'{metric}_mean']} +/- {self.scores[f'{metric}_std']}")

        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max

    def cross_validate(self, data) -> dict:
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
                'f1_std': 0
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
            scores[f'{metric}_mean'] = round((100 * results[f'test_{metric}'].mean()), 2)
            scores[f'{metric}_std'] = round((100 * results[f'test_{metric}'].std()), 2)

        return scores