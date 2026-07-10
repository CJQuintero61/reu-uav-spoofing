"""
random_forest_standalone.py

This script is a standalone implementation of the RF model

To run:
    python <filename>.py
"""
import time
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split, StratifiedGroupKFold

# testing and metric constants
TEST_SIZE = 0.20
SEED = 0
ROUND_PRECISION = 4
SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'matthews_corrcoef']
FILE = '../../../data/all_run_datasets.csv'

"""if using colab"""
# from google.colab import drive
# drive.mount('/content/drive')
# file = '/content/drive/MyDrive/all_run_datasets.csv'

class Data():
    """
    This data class reads and prepares the data for training and testing a model.
    """
    def __init__(self, file):
        """
        Args:
            file (str): path to the dataset CSV file
        """
        self.X_Train = None
        self.X_Test = None
        self.Y_Train = None
        self.Y_Test = None
        self.dataset = None
        self.file = file
        self.groups = None
    

    def read_file(self):
        """
        reads the file and checks for NaNs and ensures the dataset is a pandas DataFrame
        """
        self.dataset = pd.read_csv(self.file)
        assert isinstance(self.dataset, pd.DataFrame), "dataset is not a pandas DataFrame"

        nan_count = self.dataset.isnull().sum().sum()
        assert nan_count == 0, "NaNs were found in the data"
    

    def data_clean(self):
        """
        cleans the data by removing unnecessary columns and special characters from column names
        """
        columns_to_drop = []

        # removes columns that only have 1 unique value
        for col in self.dataset.columns:
            if self.dataset[col].nunique() == 1 and col != 'label':
                columns_to_drop.append(col)
                print(f'Removing column {col}: constant column')
        
        #drop timestamp as well.
        columns_to_drop.append('gps_timestamp')
        print('Removing column timestamp')

        #Drops these items from the dataset
        self.dataset.drop(columns=columns_to_drop, inplace=True)
        print(f'Removed {len(columns_to_drop)} columns')

        #Remove special symbols from colums
        #Remove the [], _, and <
        self.dataset.columns = (
            self.dataset.columns
            .str.replace("[", "_", regex=False)
            .str.replace("]", "_", regex=False)
            .str.replace("<", "_", regex=False)
        )


    def split_group_data(self):
        """
        splits the data into training and testing sets

        y = 0 for normal, 1 for spoof/malicious
        x drops the columns label and saves the features.
        """
        y = self.dataset['label']
        self.groups = self.dataset['run id']
        x = self.dataset.drop(columns=['label', 'gps condition',
                                       'run id', 'mission type',
                                       'location name'])

        print(f'Dropping: {["label", "gps condition", "run id", "mission type", "location name"]} from Training/Testing sets')

        assert 'gps condition' not in x.columns, "gps condition was not dropped successfully"
        assert 'label' not in x.columns, "label was not dropped successfully"

        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        train_idx, test_idx = next(
            sgkf.split(x, y, groups=self.groups)
        )

        self.X_Train = x.iloc[train_idx]
        self.X_Test  = x.iloc[test_idx]

        self.Y_Train = y.iloc[train_idx]
        self.Y_Test  = y.iloc[test_idx]


class RandomForestModel():
    
    def __init__(self):
        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(
            n_estimators = 100,
            random_state = SEED,
            class_weight = 'balanced'
        )

        self.sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )
    
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
            groups=data.groups.iloc[data.X_Train.index],
            cv=self.sgkf,
            scoring=SCORING
        )

        for metric in SCORING:
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


if __name__ == "__main__":
    # set up the data
    data = Data(FILE)
    data.read_file()
    data.data_clean()
    data.split_group_data()

    train_runs = set(data.groups.iloc[data.X_Train.index])
    test_runs = set(data.groups.iloc[data.X_Test.index])

    print(f"Training runs ({len(train_runs)}): {sorted(train_runs)}")
    print(f"Testing runs ({len(test_runs)}): {sorted(test_runs)}")

    assert train_runs.isdisjoint(test_runs), \
        "Run leakage detected!"

    print('\n----- Data Stats -----')
    print(f'Data shape: {data.dataset.shape}')
    print(f'Normal label count: {data.dataset["label"].value_counts()[0]}')
    print(f'Spoof label count: {data.dataset["label"].value_counts()[1]}')
    print(f'\nData columns: {data.dataset.columns}\n')
    print(f'Training set shape: {data.X_Train.shape}')
    print(f'Testing set shape: {data.X_Test.shape}')

    for col in data.X_Train.columns:
        assert col in data.X_Test.columns, f"Column {col} exists in Training set but not in Testing set"

    for col in data.X_Test.columns:
        assert col in data.X_Train.columns, f"Column {col} exists in Testing set but not in Training set"

    assert data.X_Train.shape[1] == data.X_Test.shape[1], "Training and Testing sets have different number of features"

    print(f'\nTraining/Testing set columns: {data.X_Train.columns}\n')
    print(f'Training set label distribution:\n{data.Y_Train.value_counts(normalize=True)}\n')
    print(f'Testing set label distribution:\n{data.Y_Test.value_counts(normalize=True)}\n')
    

    rf = RandomForestModel()

    print(f'\nBegin training at {time.strftime("%H:%M:%S", time.localtime())}')
    rf.train_model(data)
    print('Training complete\n')

    print(f'Begin cross-validation at {time.strftime("%H:%M:%S", time.localtime())}')
    rf.cross_validate(data)
    print('Cross-validation complete\n')

    print(f'Begin prediction at {time.strftime("%H:%M:%S", time.localtime())}')
    rf.predict(data)
    print('Prediction complete\n')

    rf.evaluate(data)
    rf.print_model_info()
    print(f"\n----------------------------------------------------------------\n")