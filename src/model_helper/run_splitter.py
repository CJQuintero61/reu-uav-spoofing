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

        #In each item in the grouped_runs.
        #Look at the values in label if == 1 put it into spoof list.
        for item in grouped_runs:
            run_rows = self.read_flight_data.dataset[
                self.read_flight_data.dataset['run id'] == item
            ]
            if 1 in run_rows["label"].values:
                spoof_list.append(item)
            else:
                normal_list.append(item)
        
        #Shuffle the lists.
        random.shuffle(normal_list)
        random.shuffle(spoof_list)

        #Split the list
        split_normal = int(len(normal_list) * 0.8)
        split_spoof = max(1, int(len(spoof_list) * 0.8))

        #Ensure that there is at least one spoof in the list
        if split_spoof == len(spoof_list) and len(spoof_list) > 1:
            split_spoof -= 1

        #Put the normal and spoof 80 and 20 into it's split lists
        normal_80 = normal_list[:split_normal]
        normal_20 = normal_list[split_normal:]

        spoof_80 = spoof_list[:split_spoof]
        spoof_20 = spoof_list[split_spoof:]

        #Add together the two lists together
        training_data = normal_80 + spoof_80
        testing_data = normal_20 + spoof_20

        #Look into the dataset and find the set as testing and training.
        training = self.read_flight_data.dataset[
            self.read_flight_data.dataset['run id'].isin(training_data)
        ]

        testing = self.read_flight_data.dataset[
            self.read_flight_data.dataset['run id'].isin(testing_data)
        ]
        

        #Assigned the train and testing labels x and y.
        self.read_flight_data.X_Train = training.drop(columns=['label', 'gps condition',
                                       'run id', 'mission type',
                                       'location name'], errors='ignore')
        self.read_flight_data.Y_Train = training['label']

        self.read_flight_data.X_Test = testing.drop(columns=['label', 'gps condition',
                                       'run id', 'mission type',
                                       'location name'], errors='ignore')
        self.read_flight_data.Y_Test = testing['label']
        
        #See on terminal the training and testing runs.
        print("Training runs:")
        print(training_data)

        print()

        print("Testing runs:")
        print(testing_data)

