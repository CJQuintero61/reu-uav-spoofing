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

        #default module numbers
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        
    def fit(self, data):
        self.model_fit = self.xgb_model.fit(data.X_Train, data.Y_Train)
    
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