#imports
from read_data import ReadFlightData
from factories import ModelFactory, ActionFactory

#Client code
class ClientController():
    #Read and establish the data.
    def __init__(self):
        #Class instances
        self.read_flight_data = ReadFlightData()
        self.model_factory = ModelFactory()
        self.action_factory = ActionFactory()

        #Data variables to be used in class
        #Clean and split the data
        self.read_flight_data.data_clean()
        self.data = self.read_flight_data.split_random_data()
        self.x_train = self.read_flight_data.X_Train
        self.x_test = self.read_flight_data.X_Test
        self.y_train = self.read_flight_data.Y_Train
        self.y_test = self.read_flight_data.Y_Test

        #Num_features = the features
        #Shape gives the number of feature columns
        self.num_features = self.x_train.shape[1]

        #Only two identifies 0 = real 1 = malicious (if spoof and jam = 3)
        #nuuique gives the number of unique element counts.
        self.num_classes = self.y_train.nunique()
        #How many times the model runs the training process.
        self.epochs = 30

    def client_code(self):
        #Change the model type in the model = self.model_factory 
        #to do the different types of models.

        #Change name of model to one of the following
        #mlp, 1d, xgboost, lstm
        model = self.model_factory.create("1d", self.num_features, self.num_classes, self.epochs)
        train = self.action_factory.create("train")
        test = self.action_factory.create("test")
        evaluate = self.action_factory.create("evaluate")
        
        train.execute(model, self.read_flight_data)
        test.execute(model, self.read_flight_data)
        evaluate.execute(model, self.read_flight_data)
    
if __name__ == "__main__":
    #Current metrics with old data
    #MLP = 100% all areas
    #XG Boost = 100% all areas
    #ISTM = 86% and under
    #1D = 86% and under

    """
    New metrics with sim data (Ran on my Machine)
    MLP:
      Accuracy: 0.9691709386600672
      Precision: 0.968681149991942
      Recall: 0.9691709386600672
      F1: 0.967836723771585
      Confusion: [[74537 423]]
                 [[2193 7702]]

    XG Boost:
      Accuracy: 0.9949207471569147
      Precision: 0.9949125444570706
      Recall: 0.9949207471569147
      F1: 0.9948875354730531
      Confusion: [[74891 69]]
                 [[362 9533]]
    
    ISTM:
      Accuracy: 0.8833755672107961
      Precision: 0.7803523927449959
      Recall: 0.8833755672107961
      F1: 0.8286742233793193
      Confusion: [[74950 0]]
                 [[9895 0]]
    
    1D:
      Accuracy: 0.8833637810124344
      Precision: 0.7803511784791142
      Recall: 0.8833637810124344
      F1: 0.8286683528403065
      Confusion: [[74949 1]]
                 [[9895 0]]
    """

    """
    New metrics with sim data (Ran on my Machine)
    MLP:
      Accuracy: 
      Precision: 
      Recall: 
      F1: 
      Confusion: [[ ]]
                 [[ ]]

    XG Boost:
      Accuracy: 
      Precision: 
      Recall: 
      F1: 
      Confusion: [[ ]]
                 [[ ]]
    
    ISTM:
      Accuracy: 
      Precision: 
      Recall: 
      F1: 
      Confusion: [[ ]]
                 [[ ]]
    
    1D:
      Accuracy: 
      Precision: 
      Recall: 
      F1: 
      Confusion: [[ ]]
                 [[ ]]
    """

    print("client started")
    client = ClientController()
    client.client_code()