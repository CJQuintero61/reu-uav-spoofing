#Model abtract
from drone_basics.abstracts import AbstractModel

"""
Module for mlp modle implementation
To use elements from the abstract method
"""

#Imports for models
from sklearn.neural_network import MLPClassifier #MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MLPModel(AbstractModel):
    def __init__(self):
        super().__init__()
        
        #model with default parameters
        self.mlp_class = MLPClassifier()

    def fit(self, data):
        data.scale_data()
        self.model_fit = self.mlp_class.fit(data.X_Train, data.Y_Train)
    
    def predict(self, data):
        self.model_prediction = self.model_fit.predict(data.X_Test)
        return self.model_prediction
    
    def evaluate(self, data):
        self.accuracy = accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)

        return self.accuracy, self.precision, self.recall, self.f1, self.confussion_max