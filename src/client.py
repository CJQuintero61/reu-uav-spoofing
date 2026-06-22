#imports
from drone_basics.read_data import ReadFlightData
from drone_basics.factories import ModelFactory, ActionFactory

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
        model = self.model_factory.create("1d", self.num_features, self.num_classes, self.epochs)
        train = self.action_factory.create("train")
        test = self.action_factory.create("test")
        evaluate = self.action_factory.create("evaluate")
        
        train.execute(model, self.read_flight_data)
        test.execute(model, self.read_flight_data)
        evaluate.execute(model, self.read_flight_data)
    
if __name__ == "__main__":
    #Current metrics
    #MLP = 100% all areas
    #XG Boost = 100% all areas
    #ISTM = 86% and under
    #1D = 86% and under
    print("client started")
    client = ClientController()
    client.client_code()