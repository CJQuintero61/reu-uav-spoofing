"""
XG Boost code for one of our models for the classification models

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

#Import for abstract and supports
from drone_basics.abstracts import AbstractModel

#Import for models
import xgboost as xgb #XGBoost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#XGBoost model implemented from the XGBoost library
#Read and initilize the abstract connection
class XGBoostModel(AbstractModel):
    def __init__(self):
        super().__init__()

        #model with default parameters = 100% nope
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        
    def train_model(self, data):
        print("Train Called\n")
        self.model_fit = self.xgb_model.fit(data.X_Train, data.Y_Train)
        return self.model_fit
    
    def predict(self, data):
        print("Predict Called\n")
        self.model_prediction = self.model_fit.predict(data.X_Test)
        return self.model_prediction
    
    def evaluate(self, data):
        print("valuate Called")
        self.accuracy = accuracy_score(data.Y_Test, self.model_prediction)
        self.precision = precision_score(data.Y_Test, self.model_prediction, average="weighted")
        self.recall = recall_score(data.Y_Test, self.model_prediction, average="weighted")
        self.f1 = f1_score(data.Y_Test, self.model_prediction, average="weighted")
        self.confussion_max = confusion_matrix(data.Y_Test, self.model_prediction)

#Print out the model matrics
        print(
            f"Model XG Boost Accuracy: {self.accuracy}\n",
            f"Model XG Boost Precision: {self.precision}\n",
            f"Model XG Boost Recall: {self.recall}\n",
            f"Model XG Boost F1: {self.f1}\n",
            f"Model XG Boost Confusion: {self.confussion_max}"
        )