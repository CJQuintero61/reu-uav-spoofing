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
from abstracts import AbstractModel

#Import for models
import xgboost as xgb #XGBoost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle, os, time

#XGBoost model implemented from the XGBoost library
#Read and initilize the abstract connection
class XGBoostModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
        self.n_est = config["n_estimators"]
        self.learn_rate = config["learning_rate"]
        self.max_dep = config["max_depth"] 
        self.scale_weight = config["scale_pos_weight"]

        #model with default parameters = 100% nope
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_est,
            learning_rate=self.learn_rate,
            max_depth=self.max_dep,
            scale_pos_weight=self.scale_weight

        )
        
    def train_model(self, data):
        print("Train Called\n")
        start_time = time.perf_counter()

        self.model_fit = self.xgb_model.fit(data.X_Train, data.Y_Train)

        #Save the model size
        pickle.dump(self.model_fit, open("xgboost.pkl", "wb"))
        self.model_size = os.path.getsize("xgboost.pkl") / 1024

        #Get the time
        self.training_time = time.perf_counter() - start_time

        return self.model_fit
    
    def predict(self, data):
        print("Predict Called\n")
        start_time = time.perf_counter()

        self.model_prediction = self.model_fit.predict(data.X_Test)
        
        #Get the time for testing
        self.testing_time = time.perf_counter() - start_time
        
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
            f"\nModel XG Boost Accuracy: {self.accuracy}\n",
            f"Model XG Boost Precision: {self.precision}\n",
            f"Model XG Boost Recall: {self.recall}\n",
            f"Model XG Boost F1: {self.f1}\n",
            f"Model XG Boost Confusion: {self.confussion_max}\n"
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

