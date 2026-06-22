#Import for abstract
from drone_basics.abstracts import AbstractModel
from drone_basics.read_data import ReadFlightData

#LSTM imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

#Data reader helper
from drone_basics.window_module import WindowingModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#LSTM Model implemented using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        #Sets up LSTm 
        self.lstm = nn.LSTM(
            num_features,
            hidden_size=64,
            batch_first=True
        )

        self.full_connected = nn.Linear(64, num_classes)
    
    def forward(self, x):
        #returned values from the model
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1] #remove the layer to get only batch and hidden_size to work with linear
        x = self.full_connected(x) #turns the lstm representaiton to prediction.
        return x

class LSTMExecution(AbstractModel):
    def __init__(self, num_features, num_classes, epochs):
        super().__init__()

        self.lstm_model = LSTMModel(num_features, num_classes)
        self.num_features = num_features
        self.num_classes = num_classes
        self.window_module = WindowingModule(10)
        self.epochs = epochs

    #Training section
    def fit(self, data):
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
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.0001)
        
        #Train
        self.lstm_model.train()
        for epoch in range(self.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    #predict section (Currently uses the test data. Noting feed yet)
    def predict(self, data):
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
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

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

        return self.predications, self.y_test_window

    def evaluate(self, data):
        self.accuracy = accuracy_score(self.y_test_window, self.predications)
        self.precision = precision_score(self.y_test_window, self.predications, average="weighted")
        self.recall = recall_score(self.y_test_window, self.predications, average="weighted")
        self.f1 = f1_score(self.y_test_window, self.predications, average="weighted")
        self.confussion_max = confusion_matrix(self.y_test_window, self.predications)

        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max 