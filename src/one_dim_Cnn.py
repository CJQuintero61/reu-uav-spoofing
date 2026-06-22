"""
1D CNN code for one of our models for the classification models

This is 1 dimensional CNN model that uses a sliding window to read
and analyze the data.

The pipeline for this specific file:
-> Def model_name
  1) First is the neural structure for the CNN
  
-> Def train_model()
  1) The model is created in __init__() along with
     the instances for the sliding window script and
     parameters for the model.
  2) The data is first scaled
  3) It is then processed and seperated into a sliding window
     format.
  4) It is then transposed.
  5) Runs the data into the model at epochs times.

-> Def Predict()
  1) Run the sliding window on the test data.
  2) Transpose
  3) Convert to tensor
  4) Evaluate from the models
  5) Run the models on the test data (changes it to numpy array)
  6) Return the predications and Y_Test.

-> Def Evaluate()
  1) Get the scores for all metrics.
  2) Print mertrics.
"""

#Import for abstract
from drone_basics.abstracts import AbstractModel

#1D-CNN model imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

#Data reader helper
from drone_basics.window_module import WindowingModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Model 1D CNN implemented using PyTorch
class OneDimCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            #(in_channels:# Features, out_channels:# Filters, kernel:# examine at one)
            nn.Conv1d(num_features, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(2) #Reduces sequence length
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64), #Finds the linear automaticly.
            nn.ReLU(),
            nn.Linear(64, num_classes) #Determin on the final layer if spoof or not (Note: change to 3 if doing three)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class OneDimExecution(AbstractModel):
    def __init__(self, num_features, num_classes, epochs):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.window_module = WindowingModule(10)
        self.one_dim_model = OneDimCNN(num_features, num_classes)
        self.epochs = epochs

    #Training section
    def train_model(self, data):
        #Scale and change dataset to a sliding window
        data.scale_data()
        self.x_window, self.y_window = (
            self.window_module.create_window(
            data.X_Train, data.Y_Train)
        )
        
        #Transpose flips the parameters to the correct format
        #Don't need to do the labels (Y)
        self.transpose_x_win = np.transpose(self.x_window, (0, 2, 1))

        #Convert to Tensor
        x_np_array = np.array(self.transpose_x_win)
        x_tensor_data = torch.from_numpy(x_np_array).float()
        y_tensor_data = torch.from_numpy(self.y_window).long()
        dataset = TensorDataset(x_tensor_data, y_tensor_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.one_dim_model.parameters(), lr=0.0001)

        self.one_dim_model.train()
        #Train
        for epoch in range(self.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.one_dim_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()


    def predict(self, data):
         #Change dataset to a sliding window
        self.x_window, self.y_window = (
            self.window_module.create_window(
            data.X_Test, data.Y_Test)
        )
        
        #Transpose flips the parameters to the correct format
        #Don't need to do the labels (Y)
        self.transpose_x_win = np.transpose(self.x_window, (0, 2, 1))

        #Convert to Tensor
        x_np_array = np.array(self.transpose_x_win)
        x_tensor_data = torch.from_numpy(x_np_array).float()
        y_tensor_data = torch.from_numpy(self.y_window).long()
        dataset = TensorDataset(x_tensor_data, y_tensor_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.one_dim_model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                outputs = self.one_dim_model(x_batch)
                predications = torch.argmax(outputs, dim=1).numpy()
                
                all_preds.extend(predications)
                all_labels.extend(y_batch.numpy())
        
                self.predications = all_preds
                self.y_test_window = all_labels
        return self.predications, self.y_test_window

    def evaluate(self, data):
        self.accuracy = accuracy_score(self.y_test_window, self.predications)
        self.precision = precision_score(self.y_test_window, self.predications, average="weighted")
        self.recall = recall_score(self.y_test_window, self.predications, average="weighted")
        self.f1 = f1_score(self.y_test_window, self.predications, average="weighted")
        self.confussion_max = confusion_matrix(self.y_test_window, self.predications)

        #Print out the model matrics
        print(
            f"\nModel 1D CNN Accuracy: {self.accuracy}\n",
            f"Model 1D CNN Precision: {self.precision}\n",
            f"Model 1D CNN Recall: {self.recall}\n",
            f"Model 1D CNN F1: {self.f1}\n",
            f"Model 1D CNN Confusion: {self.confussion_max}"
        )