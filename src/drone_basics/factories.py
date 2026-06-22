#import models
from drone_basics.models.lstm_model import LSTMExecution
from drone_basics.models.mlp import MLPModel
from drone_basics.models.one_dim_Cnn import OneDimExecution
from drone_basics.models.xg_boost import XGBoostModel

#import action types
from drone_basics.execute_action import TrainAction, TestAction, EvaluateAction

#Model "Factory" creates/runs the model type
class ModelFactory():    
    def create(self, model_type, num_features, num_classes, epochs):
        #A dictory that matchs the string variables
        #Avoids using if else and cases
        model_type = model_type.lower().strip()
        model_map = {
            #XGBoost and every case
            "xgboost" : lambda: XGBoostModel(),
            "xg boost": lambda: XGBoostModel(),

            #MLP
            "mlp": lambda: MLPModel(),

            #1D Cnn to get every case
            "1d cnn": lambda: OneDimExecution(
                num_features, num_classes, epochs
                ),
            "1d": lambda: OneDimExecution(
                num_features, num_classes, epochs
                ),
            "one dim": lambda: OneDimExecution(
                num_features, num_classes, epochs
                ),
            "one d": lambda: OneDimExecution(
                num_features, num_classes, epochs
                ),
            
            #LSTM
            "lstm": lambda: LSTMExecution(
                num_features, num_classes, epochs
                )
        }
        
        #Returns the the model type from the map
        print(f"Running model {model_type}")
        return model_map[model_type]()
        
#Factory to handle the action type that the models does
class ActionFactory():    
    def create(self, action_type):
        #Same thing with the class ModelFactory
        action_type = action_type.lower().strip()
        action_map = {
            "train" : lambda: TrainAction(),
            "test" : lambda: TestAction(),
            "evaluate" : lambda: EvaluateAction()
        }

        #Return the type from the dictorty
        print(f"Running type {action_type}")
        return action_map[action_type]()
    