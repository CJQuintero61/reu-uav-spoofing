"""
MLP code for one of our models for the classification models

Use with the abstract and factory.

The pipeline for this specific file:

-> Def train_model()
  1) Runs library fit functio

-> Def Predict()
  1) Runs library predict function

-> Def Evaluate()
  1) Get the scores for all metrics.
  2) Print mertrics.
"""

#Model abtract
from src.model_helper.abstracts import AbstractModel

#Imports for models
from sklearn.neural_network import MLPClassifier #MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import time, os, pickle

class MLPModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
        self.hid_lay_size = config["hidden_layer_sizes"]
        self.learn_rate = config["learning_rate_init"]
        self.max_it = config["max_iter"]
        
        #model with default parameters = 100% nope
        self.mlp_class = MLPClassifier(
            hidden_layer_sizes=self.hid_lay_size,
            activation="relu",
            solver="adam",
            max_iter=self.max_it,
            learning_rate_init=self.learn_rate
        )

    def train_model(self, data):
        start_time = time.perf_counter()
        data.scale_data()
        self.model_fit = self.mlp_class.fit(data.X_Train, data.Y_Train)
        
        #Save the model size
        pickle.dump(self.model_fit, open("mlp.pkl", "wb"))
        self.model_size = os.path.getsize("mlp.pkl") / 1024

        #Get the time
        self.training_time = time.perf_counter() - start_time

    def predict(self, data):
        start_time = time.perf_counter()
        self.model_prediction = self.model_fit.predict(data.X_Test)
        
        #Get the time for testing
        self.testing_time = time.perf_counter() - start_time
        
        return self.model_prediction
    
    def evaluate(self, data):
        self.accuracy = accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)

        #Print out the model matrics
        print(
            f"Model MLP Accuracy: {self.accuracy}\n",
            f"Model MLP Precision: {self.precision}\n",
            f"Model MLP Recall: {self.recall}\n",
            f"Model MLP F1: {self.f1}\n",
            f"Model MLP Confusion: {self.confussion_max}\n"
        )

        self.precision_per_class = precision_score(
            data.Y_Test,
            self.model_prediction,
            average=None,
            zero_division=0
        )

        self.recall_per_class = recall_score(
            data.Y_Test,
            self.model_prediction,
            average=None,
            zero_division=0
        )

        print(f"Precision per class: {self.precision_per_class}\n")
        print(f"Recall per class: {self.recall_per_class}\n")

        #Print computional meterics
        print(
            f"Training Time: {self.training_time:.4f} seconds.\n",
            f"Testing Time: {self.testing_time:.4f} seconds.\n",
            f"Model Size (KB): {self.model_size:.2f}\n"
        )

