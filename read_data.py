"""
Reads the data and cleans the data here.
Feeds this data into the models for training.

HOW TO USE:
1) Import this file like this "from read_data import ReadFlightData"
2) Create an instance of the class (readflightdata = ReadFlightData())
3) Use the instance variable with variable.function to use the function you want.
4) Create a variable like X_train = variable.X_Train, Y_train = variable.X_Train to get the training data
5) Use X_train and Y_train during model predication for model.fit()/predict()

Self doesn't take any variables when using it in an instance.
File has been tested and is working properly.
File also does not need the if __name__ == "__main__" it is for testing only.

Note with scaler:
Scaler is not needed for every model. It depends entirely on the model you are using.
It is here though if needed.

Created in WSL.
"""

#Imports
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read number flight data
# Rewriten to work with class from Cj's model 1 file (Thank you)
class ReadFlightData():
    def __init__(self):
        self.director = os.path.dirname(__file__)
        self.file_name = os.path.join(self.director, "spoofing-merged-gps-only.csv")
        self.dataset = pd.read_csv(self.file_name)
        
        nan_count = self.dataset.isnull().sum().sum()
        assert nan_count == 0, "NaNs were found in the data"

    #Cleans the data by removing the unnecessary columns from data.
    #Places them into an array.
    def data_clean(self):
        columns_to_drop = []
        for col in self.dataset.columns:
            if self.dataset[col].nunique() == 1:
                columns_to_drop.append(col)
                print(f'Removing column {col}')
        
        #drop timestamp as well.
        columns_to_drop.append('timestamp')
        print('Removing column timestamp')

        #Drops these items from the dataset
        self.dataset.drop(columns=columns_to_drop, inplace=True)
        print(f'Removed {len(columns_to_drop)} columns')

    #split the data where:
    # y = 0 for real, 1 for spoof/malicious
    # x drops the columns label and saves the features.
    def split_random_data(self):
        y = self.dataset['label'].map({'benign': 0, 'malicious': 1})
        x = self.dataset.drop(columns=['label'])

        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(
            x,
            y,
            test_size=.20,
            shuffle=True,
            random_state=0,
            stratify = y
        )

    #Note:
    # All models do not need to be scaled.
    # For the trianing data fit and transform
    # For the test data just TRANSFORM (Could create a data leak if fit and transformed)
    def scale_data(self):
        scaler = StandardScaler()
        self.X_Train = scaler.fit_transform(self.X_Train) #Analyzes and calculates the mean and cariance of the data and transforms it
        self.X_Test = scaler.transform(self.X_Test) #Transforms the data using the format from the training data


#Testing section of the code.
#If uncommented, it will only run in this file and not in any file.
"""
if __name__ == "__main__":
    readflightdata = ReadFlightData()
    readflightdata.data_clean()
    readflightdata.split_random_data()
    X_train = readflightdata.X_Train
    X_test = readflightdata.X_Test
    readflightdata.scale_data()

    print("Completed dataset cleaning session.")
    print("Training Features:\n", X_train)
    print("Testing Features:\n", X_test)

"""