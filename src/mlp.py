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
from drone_basics.abstracts import AbstractModel



#Imports for models
from sklearn.neural_network import MLPClassifier #MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MLPModel(AbstractModel):
    def __init__(self):
        super().__init__()
        
        #model with default parameters = 100% nope
        self.mlp_class = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=500
        )

    def train_model(self, data):
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

        #Print out the model matrics
        print(
            f"Model MLP Accuracy: {self.accuracy}\n",
            f"Model MLP Precision: {self.precision}\n",
            f"Model MLP Recall: {self.recall}\n",
            f"Model MLP F1: {self.f1}\n",
            f"Model MLP Confusion: {self.confussion_max}"
        )