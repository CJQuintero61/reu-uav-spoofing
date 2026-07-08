import sys
import time
import pickle
import pandas as pd
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

TEST_SIZE = 0.20
SEED = 0
ROUND_PRECISION = 4
SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'matthews_corrcoef']


C_VALUES = [0.1, 1, 10]
GAMMA_VALUES = ['scale', 0.1]

"""
create a list of parameter combinations for the SVC model

there should be 6 combinations: 3 C values * 2 gamma values
all combinations will use the 'rbf' kernel and 'balanced' class_weights
"""
PARAMS = [
    {
        'id': i,
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'C': C,
        'gamma': gamma,
    }
    for i, (C, gamma) in enumerate(
        itertools.product(C_VALUES, GAMMA_VALUES),
        start=1
    )
]

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


    def split_random_data(self):
        """
        splits the data into training and testing sets

        y = 0 for normal, 1 for spoof/malicious
        x drops the columns label and saves the features.
        """
        y = self.dataset['label']
        x = self.dataset.drop(columns=['label', 'gps condition',
                                       'run id', 'mission type',
                                       'location name'])

        print(f'Dropping: {["label", "gps condition", "run id", "mission type", "location name"]} from Training/Testing sets')

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

        self.X_Train = X_Train
        self.X_Test = X_Test
        self.Y_Train = Y_Train
        self.Y_Test = Y_Test


class SVCModel():
    """
    SVC model for classifying spoofing and benign flight data.

    NOTE: this model uses a pipeline to scale the data before fitting the SVC model.
    DO NOT call ReadFlightData.scale_data() before fitting this model, as the pipeline will
    handle scaling the data for each fold in the cross validation, and for the final fit and predict.
    """

    def __init__(self, kernel='rbf', class_weight='balanced', C=1.0, gamma='scale', random_state=SEED):
        
        # pipeline for scaling and SVC model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel = kernel,
                class_weight = class_weight,
                C = C,
                gamma = gamma,
                random_state = random_state
            ))
        ])

        # Stratified K-Fold cross-validator
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
    

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

        
        self._set_model_size()
        
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
            scoring=SCORING
        )

        for metric in SCORING:
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
        print(f"Model Size:         {round(self.model_size_kb, ROUND_PRECISION)} KB")
        print(f"Training Time:      {round(self.train_time, ROUND_PRECISION)} seconds")
        print(f"Prediction Time:    {round(self.predict_time, ROUND_PRECISION)} seconds")

        print("\nSVC Model Evaluation:")
        print(f"SVC Accuracy:  {round(self.accuracy, ROUND_PRECISION)}")
        print(f"SVC Precision: {round(self.precision, ROUND_PRECISION)}")
        print(f"SVC Recall:    {round(self.recall, ROUND_PRECISION)}")
        print(f"SVC F1:        {round(self.f1, ROUND_PRECISION)}")
        print(f"SVC MCC:       {round(self.mcc, ROUND_PRECISION)}")
        print(f"SVC Confusion Matrix:\n{self.confussion_max}")

        print("\nCross Validation Scores:")
        for metric in SCORING:
            print(f"{metric} mean: {round(self.cv_scores[f'{metric}_mean'], ROUND_PRECISION)}")
            print(f"{metric} std:  {round(self.cv_scores[f'{metric}_std'], ROUND_PRECISION)}\n")

class Tee:
    """
    Writes everything to both the terminal and a file at once
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    # data file to read
    file = '../../../data/imbalanced_data_80_20.csv'

    # change the imbalance ratio when using different datasets
    output_base = '../../../data/svc_model_80_20'
    output = f'{output_base}_results.txt'
    summary_csv = f'{output_base}_summary.csv'

    original_stdout = sys.stdout
    results = []  # will hold one dict per parameter combo

    with open(output, 'w') as f:
        sys.stdout = Tee(original_stdout, f)

        try:
            # set up the data
            data = Data(file)
            data.read_file()
            data.data_clean()
            data.split_random_data()

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

            for idx, param in enumerate(PARAMS):
                print(f"\n----------------------------------------------------------------\n")
                print(f"\nRunning parameter combination {idx + 1}/{len(PARAMS)}")
                print(f"\n----- Training SVC model with parameters: {param} -----")

                svc = SVCModel(
                    kernel=param['kernel'],
                    class_weight=param['class_weight'],
                    C=param['C'],
                    gamma=param['gamma'],
                )

                print(f'\nBegin training at {time.strftime("%H:%M:%S", time.localtime())}')
                svc.train_model(data)
                print('Training complete\n')

                print(f'Begin cross-validation at {time.strftime("%H:%M:%S", time.localtime())}')
                svc.cross_validate(data)
                print('Cross-validation complete\n')

                print(f'Begin prediction at {time.strftime("%H:%M:%S", time.localtime())}')
                svc.predict(data)
                print('Prediction complete\n')

                svc.evaluate(data)
                svc.print_model_info()
                print(f"\n----------------------------------------------------------------\n")

                # collect this run's results into a flat dict for the summary CSV
                row = {
                    'id': param['id'],
                    'kernel': param['kernel'],
                    'class_weight': param['class_weight'],
                    'C': param['C'],
                    'gamma': param['gamma'],
                    'accuracy': round(svc.accuracy, ROUND_PRECISION),
                    'precision': round(svc.precision, ROUND_PRECISION),
                    'recall': round(svc.recall, ROUND_PRECISION),
                    'f1': round(svc.f1, ROUND_PRECISION),
                    'mcc': round(svc.mcc, ROUND_PRECISION),
                    'model_size_kb': round(svc.model_size_kb, ROUND_PRECISION),
                    'train_time_s': round(svc.train_time, ROUND_PRECISION),
                    'predict_time_s': round(svc.predict_time, ROUND_PRECISION),
                }
                # add the cross-validation mean/std for each metric too
                for metric in SCORING:
                    row[f'cv_{metric}_mean'] = round(svc.cv_scores[f'{metric}_mean'], ROUND_PRECISION)
                    row[f'cv_{metric}_std'] = round(svc.cv_scores[f'{metric}_std'], ROUND_PRECISION)

                results.append(row)

        finally:
            sys.stdout = original_stdout

    # build the summary DataFrame, sort by MCC (best first), and save
    summary_df = pd.DataFrame(results)
    summary_df.sort_values(by='mcc', ascending=False, inplace=True)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nSummary written to {summary_csv}")
    print(summary_df)