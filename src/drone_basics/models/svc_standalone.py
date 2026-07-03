import os
import time
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

TEST_SIZE = 0.20
SEED = 0

"""if using colab"""
# from google.colab import drive
# drive.mount('/content/drive')
# file = '/content/drive/MyDrive/all_run_datasets.csv'


file = '../../../data/all_run_datasets.csv'
dataset = pd.read_csv(file)

nan_count = dataset.isnull().sum().sum()
assert nan_count == 0, "NaNs were found in the data"

# Cleans the data by removing the unnecessary columns from data.
def data_clean():
    columns_to_drop = []
    
    for col in dataset.columns:
        # remove columns that have only 1 unique value
        if dataset[col].nunique() == 1 and col != 'label':
            columns_to_drop.append(col)
            print(f'Removing column {col}: constant column')
            
        # remove columns that have string values
        if isinstance(dataset[col][1], str) and col != 'label':
            columns_to_drop.append(col)
            print(f'Removing column {col}: string column')

    # Drops these items from the dataset
    dataset.drop(columns=columns_to_drop, inplace=True)
    print(f'Removed {len(columns_to_drop)} columns')

    # Remove special symbols from colums
    # Remove the [], _, and <
    dataset.columns = (
        dataset.columns
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
    )

# cleans the data inplace
data_clean()

# split the data where:
# y = 0 for real, 1 for spoof/malicious
# x drops the columns label and saves the features.
def split_random_data():
    y = dataset['label']
    x = dataset.drop(columns=['label'])

    assert 'gps condition' not in x.columns, "gps condition was not dropped successfully"
    assert 'label' not in x.columns, "label was not dropped successfully"

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=SEED,
        stratify = y
    )

    return X_Train, X_Test, Y_Train, Y_Test

X_Train, X_Test, Y_Train, Y_Test = split_random_data()

class Data():
    def __init__(self, X_Train, X_Test, Y_Train, Y_Test):
        self.X_Train = X_Train
        self.X_Test = X_Test
        self.Y_Train = Y_Train
        self.Y_Test = Y_Test

data = Data(X_Train, X_Test, Y_Train, Y_Test)


class SVCModel():
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

        
        self._set_model_size()  # calculate the model size in KB
        self.print_model_info()
        
        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max

    def cross_validate(self, data):
        """
        perform cross validation using the pipeline and the stratified k-fold cross-validator
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

        self.cv_scores = scores


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

svc = SVCModel()

print('starting model training')
svc.train_model(data)
print('model training complete')

print('starting cross validation')
svc.cross_validate(data)
print('cross validation complete')

print('starting model prediction')
svc.predict(data)
print('model prediction complete')

print('starting model evaluation')
svc.evaluate(data)
print('model evaluation complete\n')


svc.print_model_info()