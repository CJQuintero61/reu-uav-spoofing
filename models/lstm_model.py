"""
ISTM code for one of our models for the classification models

Use with the abstract and factory.

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
from abstracts import AbstractModel

#LSTM imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

#Data reader helper
from window_module import WindowingModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time 
import os

#LSTM Model implemented using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes, config):
        super().__init__()
        #Sets up LSTm 
        self.lstm = nn.LSTM(
            num_features,
            hidden_size=config["hidden_size"],
            batch_first=True
        )

        self.full_connected = nn.Linear(config["hidden_size"], num_classes)
    
    def forward(self, x):
        #returned values from the model
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1] #remove the layer to get only batch and hidden_size to work with linear
        x = self.full_connected(x) #turns the lstm representaiton to prediction.
        return x

class LSTMExecution(AbstractModel):
    def __init__(self, num_features, num_classes, config):
        super().__init__()

        self.lstm_model = LSTMModel(num_features, num_classes, config)
        self.num_features = num_features
        self.num_classes = num_classes
        self.window_module = WindowingModule(config["window_size"])
        self.epochs = config["epochs"]
        self.learn_rate = config["learning_rate"]
        self.bat_size = config["batch_size"]

    #Training section
    def train_model(self, data):
        start_time = time.perf_counter()
        #Scale and change dataset to a sliding window
        data.scale_data()
        self.x_window, self.y_window = (
            self.window_module.create_window(
            data.X_Train, data.Y_Train)
        )

        #Convert to Tensor
        x_np_array = np.array(self.x_window)
        x_tensor_data = torch.from_numpy(x_np_array).float()
        y_tensor_data = torch.from_numpy(self.y_window).long()
        dataset = TensorDataset(x_tensor_data, y_tensor_data)
        
        #load data, assess and optimizes erros
        loader = DataLoader(dataset, batch_size=self.bat_size, shuffle=True)
        class_weight = torch.tensor([1.0, 4.0]).float()
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learn_rate)
        
        #Train
        self.lstm_model.train()
        for epoch in range(self.epochs):
            total_loss = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")
        
        #Save the model size        
        self.training_time = time.perf_counter() - start_time
        torch.save(
            self.lstm_model.state_dict(),
            "lstm.pth"
        )

        self.model_size = os.path.getsize("lstm.pth") / 1024
    

    #predict section (Currently uses the test data. Noting feed yet)
    def predict(self, data):
        start_time = time.perf_counter()
        #Change dataset to a sliding window
        self.x_window, self.y_window = (
            self.window_module.create_window(
            data.X_Test, data.Y_Test)
        )
        
        #Convert to Tensor
        x_np_array = np.array(self.x_window)
        x_tensor_data = torch.from_numpy(x_np_array).float()
        y_tensor_data = torch.from_numpy(self.y_window).long()
        dataset = TensorDataset(x_tensor_data, y_tensor_data)
        loader = DataLoader(dataset, batch_size=self.bat_size, shuffle=False)

        #Evaluates and gives teh metrics for the model later
        self.lstm_model.eval()

        all_preds = []
        all_labels = []

        #Used to predict using the trained model
        with torch.no_grad():
            for x_batch, y_batch in loader:
                outputs = self.lstm_model(x_batch)
                predications = torch.argmax(outputs, dim=1).numpy()
                
                all_preds.extend(predications)
                all_labels.extend(y_batch.numpy())
        
                self.predications = all_preds
                self.y_test_window = all_labels
        
        self.testing_time = time.perf_counter() - start_time
        return self.predications, self.y_test_window

    def evaluate(self, data):
        self.accuracy = accuracy_score(self.y_test_window, self.predications)
        self.precision = precision_score(self.y_test_window, self.predications, average="weighted")
        self.recall = recall_score(self.y_test_window, self.predications, average="weighted")
        self.f1 = f1_score(self.y_test_window, self.predications, average="weighted")
        self.confussion_max = confusion_matrix(self.y_test_window, self.predications)
        
        #Calculate the precision and recall per class
        self.precision_per_class = precision_score(
            self.y_test_window,
            self.predications,
            average=None
        )

        self.recall_per_class = recall_score(
            self.y_test_window,
            self.predications,
            average=None
        )

        #Print out the model matrics
        print(
            f"Model LSTM Accuracy: {self.accuracy}\n",
            f"Model LSTM Precision: {self.precision}\n",
            f"Model LSTM Recall: {self.recall}\n",
            f"Model LSTM F1: {self.f1}\n",
            f"Model LSTM Confusion: {self.confussion_max}\n"
        )

        #Prints for precision and recall
        print(f"precision per class: {self.precision_per_class}\n")
        print(f"Recall per class: {self.recall_per_class}\n")

        #Print computional meterics
        print(
            f"Training Time: {self.training_time:.4f} seconds.\n",
            f"Testing Time: {self.testing_time:.4f} seconds.\n",
            f"Model Size (KB): {self.model_size:.2f}\n"
        )
        
        