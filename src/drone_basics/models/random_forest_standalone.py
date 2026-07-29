"""
random_forest_standalone.py

This script is a standalone implementation of the RF model

NOTE: This script was made specifically for the real_spoofed_data.csv. The model performs
perfectly with 100% accuracy, precision, recall, F1, and MCC. The learning curve for F1 is also produced
for analysis. This script and its results are not included or referenced in the final report or in the powerpoint.
"""

import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.base import clone


# testing and metric constants
TEST_SIZE = 0.20
SEED = 0
ROUND_PRECISION = 4
FILE = '../real_spoofed_data.csv'


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
                print(f'Removing constant column {col}')
        
        #drop timestamp as well.
        columns_to_drop.append('timestamp')
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

        # convert the label column to binary values
        self.dataset['label'] = self.dataset['label'].apply(lambda x: 0 if x == 'benign' else 1)
        assert set(self.dataset['label'].unique()) == {0, 1}, "Label column conversion failed"
    

    def split_group_data(self):
        """
        splits the data into training and testing sets

        y = 0 for normal, 1 for spoof/malicious
        x drops the columns label and saves the features.
        """

        # split the data into training and testing sets
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(
            self.dataset.drop(columns=['label']),
            self.dataset['label'],
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=self.dataset['label']
        )

        assert 'label' not in self.X_Train.columns, "Label column found in training features"
        assert 'label' not in self.X_Test.columns, "Label column found in testing features"

class RandomForestModel():
    
    def __init__(self):
        # init the random forest model with 100 trees and balanced class weights for imbalanced data
        self.model = RandomForestClassifier(
            n_estimators = 100,
            max_depth = None,
            min_samples_leaf = 1,
            random_state = SEED
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


    def plot_learning_curve(self, trained_model, X_train, y_train, cv_splits=5):
        """
        Builds a learning curve using the same hyperparameters as trained_model,
        evaluated only on X_train/y_train (aka it never touches X_Test).
        """
        # clone() copies hyperparameters only - NOT the fitted model
        model = clone(trained_model)
        cv_arg = cv_splits

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv_arg,
            scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 10),
            shuffle=True,
            random_state=SEED,
            n_jobs=-1
        )

        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='tab:blue', label='Training F1')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='tab:blue')
        plt.plot(train_sizes, test_mean, 'o-', color='tab:orange', label='Cross-Validation F1')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='tab:orange')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title('Random Forest Learning Curve')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('rf_learning_curve_f1.png', dpi=200)
        plt.show()


if __name__ == "__main__":
    # set up the data
    data = Data(FILE)
    data.read_file()
    data.data_clean()
    data.split_group_data()

    print('\n----- Data Stats -----')
    print(f'Data shape: {data.dataset.shape}')
    print(f'Normal label count: {data.dataset["label"].value_counts()[0]}')
    print(f'Spoof label count: {data.dataset["label"].value_counts()[1]}')
    print(f'\nData columns: {data.dataset.columns}\n')
    print(f'Training set shape: {data.X_Train.shape}')
    print(f'Testing set shape: {data.X_Test.shape}')

    # probably a better way to do this but oh well
    for col in data.X_Train.columns:
        assert col in data.X_Test.columns, f"Column {col} exists in Training set but not in Testing set"

    for col in data.X_Test.columns:
        assert col in data.X_Train.columns, f"Column {col} exists in Testing set but not in Training set"

    assert data.X_Train.shape[1] == data.X_Test.shape[1], "Training and Testing sets have different number of features"

    print(f'\nTraining/Testing set columns: {data.X_Train.columns}\n')
    print(f'Training set label distribution:\n{data.Y_Train.value_counts(normalize=True)}\n')
    print(f'Testing set label distribution:\n{data.Y_Test.value_counts(normalize=True)}\n')
    

    rf = RandomForestModel()

    rf.train_model(data)

    rf.plot_learning_curve(
        trained_model=rf.model,   # same parameters as what actually got trained
        X_train=data.X_Train,
        y_train=data.Y_Train,
        cv_splits=5
    )

    print("\nRandom Forest Feature Importances:")
    importance = pd.Series(
        rf.model_fit.feature_importances_,
        index=data.X_Train.columns
    ).sort_values(ascending=False)

    print(importance.head(20))

    rf.predict(data)

    rf.evaluate(data)
    rf.print_model_info()
    print(f"\n----------------------------------------------------------------\n")