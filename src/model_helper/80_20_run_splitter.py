import pandas as pd
import random
from sklearn.model_selection import train_test_split

#Deep learning specific data reading module
class DataSetup():
    def __init__(self, read_flight_data):
        self.read_flight_data = read_flight_data
    
    def split_data_by_run(self):
        normal_list = []
        spoof_list = []

        grouped_runs = self.read_flight_data.dataset['run id'].unique()

        for item in grouped_runs:
            run_rows = self.read_flight_data.dataset[
                self.read_flight_data.dataset['run id'] == item
            ]
            run_labels = run_rows['label'].iloc[0]
            if run_labels == 0:
                normal_list.append(item)
            elif run_labels == 1:
                spoof_list.append(item)
        

        random.shuffle(normal_list)
        random.shuffle(spoof_list)

        split_normal = int(len(normal_list) * 0.8)
        split_spoof = int(len(spoof_list) * 0.8)

        normal_80 = normal_list[:split_normal]
        normal_20 = normal_list[split_normal:]

        spoof_80 = spoof_list[:split_spoof]
        spoof_20 = spoof_list[split_spoof:]

        training_data = normal_80 + spoof_80
        testing_data = normal_20 + spoof_20

        training = self.read_flight_data.dataset[
            self.read_flight_data.dataset['run id'].isin(training_data)
        ]

        testing = self.read_flight_data.dataset[
            self.read_flight_data.dataset['run id'].isin(testing_data)
        ]
        
        self.read_flight_data.X_Train = training.drop(columns=['label', 'gps condition',
                                       'run id', 'mission type',
                                       'location name'])
        self.read_flight_data.Y_Train = training['label']

        self.read_flight_data.X_Test = testing.drop(columns=['label', 'gps condition',
                                       'run id', 'mission type',
                                       'location name'])
        self.read_flight_data.Y_Test = testing['label']
        
        print("Training runs:")
        print(training_data)

        print()

        print("Testing runs:")
        print(testing_data)

