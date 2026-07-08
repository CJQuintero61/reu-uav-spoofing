import time
import pickle
import pandas as pd
from drone_basics.abstracts import AbstractModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split

TEST_SIZE = 0.20
SEED = 0
ROUND_PRECISION = 4
SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'matthews_corrcoef']

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

class RandomForestModel():
    

    def __init__(self):
        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(n_estimators = 100, random_state = SEED, class_weight = 'balanced')

        # Stratified K-Fold cross-validator
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
    
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


        self._set_model_size()
        self.print_model_info()

        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max
    

    def cross_validate(self, data) -> dict:
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
            scoring=SCORING
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
        print(f"Model Size:         {round(self.model_size_kb, ROUND_PRECISION)} KB")
        print(f"Training Time:      {round(self.train_time, ROUND_PRECISION)} seconds")
        print(f"Prediction Time:    {round(self.predict_time, ROUND_PRECISION)} seconds")

        print("\nRandom Forest Model Evaluation:")
        print(f"Random Forest Accuracy:  {round(self.accuracy, ROUND_PRECISION)}")
        print(f"Random Forest Precision: {round(self.precision, ROUND_PRECISION)}")
        print(f"Random Forest Recall:    {round(self.recall, ROUND_PRECISION)}")
        print(f"Random Forest F1:        {round(self.f1, ROUND_PRECISION)}")
        print(f"Random Forest MCC:       {round(self.mcc, ROUND_PRECISION)}")
        print(f"Random Forest Confusion Matrix:\n{self.confussion_max}")

        print("\nCross Validation Scores:")
        for metric in SCORING:
            print(f"{metric} Mean: {round(self.cv_scores[f'{metric}_mean'], ROUND_PRECISION)}")
            print(f"{metric} Std:  {round(self.cv_scores[f'{metric}_std'], ROUND_PRECISION)}\n")

rf = RandomForestModel()

print('starting model training')
rf.train_model(data)
print('model training complete')

print('starting cross validation')
rf.cross_validate(data)
print('cross validation complete')

print('starting model prediction')
rf.predict(data)
print('model prediction complete')

print('starting model evaluation')
rf.evaluate(data)
print('model evaluation complete\n')


rf.print_model_info()