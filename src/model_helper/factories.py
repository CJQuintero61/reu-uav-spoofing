#import models
from models.lstm_model import LSTMExecution
from models.mlp import MLPModel
from models.one_dim_Cnn import OneDimExecution
from models.xg_boost import XGBoostModel

#import action types
from execute_action import TrainAction, TestAction, EvaluateAction

#Model "Factory" creates/runs the model type
class ModelFactory():    
    def create(self, model_type, num_features, num_classes, config):
        #A dictory that matchs the string variables
        #Avoids using if else and cases
        model_type = model_type.lower().strip()
        model_map = {
            #XGBoost and every case
            "xgboost" : lambda: XGBoostModel(config=config),
            "xg boost": lambda: XGBoostModel(config=config),

            #MLP
            "mlp": lambda: MLPModel(config=config),

            #1D Cnn to get every case
            "1d cnn": lambda: OneDimExecution(
                num_features, num_classes, config=config
                ),
            "1d": lambda: OneDimExecution(
                num_features, num_classes, config=config
                ),
            "one dim": lambda: OneDimExecution(
                num_features, num_classes, config=config
                ),
            "one d": lambda: OneDimExecution(
                num_features, num_classes, config=config
                ),
            
            #LSTM
            "lstm": lambda: LSTMExecution(
                num_features, num_classes, config=config
                )
        }
        
        #Returns the the model type from the map
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
        return action_map[action_type]()
    