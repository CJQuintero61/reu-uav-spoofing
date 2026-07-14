#imports
from read_data import ReadFlightData
from factories import ModelFactory, ActionFactory
from deep_learning_handler import DataSetup

#Client code
class ClientController():
    #Read and establish the data.
    def __init__(self):
        #Class instances
        self.read_flight_data = ReadFlightData()
        self.model_factory = ModelFactory()
        self.action_factory = ActionFactory()
        self.data_deep_learning = DataSetup(self.read_flight_data)

    def client_code(self, model_type):
        """
-------------------- DATA Section --------------------       
        """
        model_type = model_type.lower().strip()
        self.read_flight_data.data_clean()
        
        #Select data preperation based on model_type
        if model_type == "mlp" or model_type == "xgboost" or model_type == "svc" or model_type == "rf":
          self.data = self.read_flight_data.split_random_data()
        elif model_type == "1d" or model_type == "lstm":
          self.data_deep_learning.split_data_by_run()

        """
                  MODEL PARAMETERS
        """
        #For all model types
        self.x_train = self.read_flight_data.X_Train
        self.y_train = self.read_flight_data.Y_Train

        #How many times the model runs the training process.
        self.epochs = 30

        #Num_features = the features
        self.num_features = self.x_train.shape[1]

        #Only two identifies 0 = real 1 = malicious
        self.num_classes = self.y_train.nunique()

        """
-------------------- MODEL SETUP --------------------       
        """
        #Change the model type in the model = self.model_factory 
        #to do the different types of models.

        #Change name of model to one of the following
        model = self.model_factory.create(
          model_type,
          self.num_features,
          self.num_classes,
          self.epochs
        )
        
        train = self.action_factory.create("train")
        test = self.action_factory.create("test")
        evaluate = self.action_factory.create("evaluate")
        
        train.execute(model, self.read_flight_data)
        test.execute(model, self.read_flight_data)
        evaluate.execute(model, self.read_flight_data)
          
    

"""
-------------------- Called HERE --------------------
"""
if __name__ == "__main__":
    #Current metrics with old data
    #MLP = 100% all areas
    #XG Boost = 100% all areas
    #ISTM = 86% and under
    #1D = 86% and under

    """

    July 8, 2026 with updated code for data processing
    of deep learning models.
    
    New metrics with sim data (Ran on my Machine)
    MLP:
      Accuracy: 0.9716928878675387
      Precision: 0.9714871994833254
      Recall: 0.9716928878675387
      F1: 0.970443115975322
      Confusion: [[74660 300]]
                 [[2102 7793]]

    XG Boost:
      Accuracy: 0.9949207471569147
      Precision: 0.9949125444570706
      Recall: 0.9949207471569147
      F1: 0.9948875354730531
      Confusion: [[74891 69]]
                 [[362 9533]]
    
    ISTM:
      Accuracy: 0.8201826045170592
      Precision: 0.6746629411570844
      Recall: 0.8201826045170592
      F1: 0.740339776039951
      Confusion: [[76806 123]]
                 [[16716 0]]
    
    1D:
      Accuracy: 0.8290195998698764
      Precision: 0.6914313664986753
      Recall: 0.8290195998698764
      F1: 0.7540001847757972
      Confusion: [[81549 269]]
                 [[16550 0]]
    """

    """
    New metrics with sim data (Ran on CJ's Machine)
    MLP:
      Accuracy: 0.9558924710045213
      Precision: 0.9608390301236452
      Recall: 0.9558924710045213
      F1: 0.9570642379135679
      Confusion: [[61946 3092]]
                 [[498 15856]]

    XG Boost:
      Accuracy: 0.9960192647926086
      Precision: 0.9960172389898826
      Recall: 0.9960192647926086
      F1: 0.9960114636702447
      Confusion: [[64961 77]]
                 [[247 16107]]
    
    ISTM:
      Accuracy: 0.5948541646275814
      Precision: 0.863561641642054
      Recall: 0.5948541646275814
      F1: 0.6280223294244778
      Confusion: [[32294 33031]]
                 [[84 16327]]
    
    1D:
      Accuracy: 0.9081995661605206
      Precision: 0.9176445552262438
      Recall: 0.9081995661605206
      F1: 0.8971015767548308
      Confusion: [[64469 1]]
                 [[7405 8800]]
    """

    print("client started")
    client = ClientController()
    #mlp, 1d, xgboost, lstm
    client.client_code("svc")