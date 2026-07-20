#imports
from read_data import ReadFlightData
from factories import ModelFactory, ActionFactory
from deep_learning_handler import DataSetup
from parameter_dic import xg_boost_para, mlp_para, lstm_para, oneD_para, rf_para, svc_para
import os

#Client code
class ClientController():
    #Read and establish the data.
    def __init__(self):
        #Class instances
        self.read_flight_data = ReadFlightData()
        self.model_factory = ModelFactory()
        self.action_factory = ActionFactory()
        self.data_deep_learning = DataSetup(self.read_flight_data)
 
    def client_code(self, model_type, run_number, configs):
        """
-------------------- DATA Section --------------------       
        """
        #Prepare the data and results for models and comparison.
        model_type = model_type.lower().strip()
        self.read_flight_data.data_clean()

        config_amount = 1
        all_config_results = []

        #Makes the folder results found in ros2_drone_ws

        folder = "results"
        filename = os.path.join(folder, f"{model_type}_results.txt")
        print(f"Saving results to: {filename}")
        
        print(f"Starting Model: {model_type}")
        #Write the results and configuration information in a .txt file
        with open(filename, "w") as file:
            file.write(
                f"Model: {model_type}\n"
                f"Repeated 80/20 Evaluation for {run_number}\n"
            )
            
            #For each configed parameter set do the following
            for config in configs:
                print(f"Configiration Number {config_amount}.")
                all_model_runs = []
                
                #For each run set data, create model and get results
                for i in range(run_number):
                    print(f"Iteration {i} out of {run_number}")
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
                    config=config
                    )
                    
                    #Call the factory to execute the giving
                    train = self.action_factory.create("train")
                    test = self.action_factory.create("test")
                    evaluate = self.action_factory.create("evaluate")
                    
                    train.execute(model, self.read_flight_data)
                    test.execute(model, self.read_flight_data)
                    evaluate.execute(model, self.read_flight_data)

                    model_run_met = {
                        "accuracy" : model.accuracy,
                        "precision" : model.precision,
                        "recall" : model.recall,
                        "f1" : model.f1,
                        "training time" : model.training_time,
                        "testing time" : model.testing_time,
                        "model size" : model.model_size
                    }

                    all_model_runs.append(model_run_met)
            
                #Get the model metrics and calucalte there averages.
                total_accuracy = 0
                total_precision = 0
                total_recall = 0
                total_f1 = 0
                total_training_time = 0
                total_testing_time = 0
                total_model_size = 0

                #In each dictory get the models metrics and add them together
                for dic in all_model_runs:
                    total_accuracy += dic["accuracy"]
                    total_precision += dic["precision"]
                    total_recall += dic["recall"]
                    total_f1 += dic["f1"]
                    total_training_time += dic["training time"]
                    total_testing_time += dic["testing time"]
                    total_model_size += dic["model size"]
                
                #Dictory that contains all the averages for the models metrics
                total_average = {
                    "Average Accuracy" : total_accuracy / len(all_model_runs),
                    "Average Precision" : total_precision / len(all_model_runs),
                    "Average Recall" : total_recall / len(all_model_runs),
                    "Average F1" : total_f1 / len(all_model_runs),
                    "Average Training Time" : total_training_time / len(all_model_runs),
                    "Average Testing Time" : total_testing_time / len(all_model_runs),
                    "Average Model Size" : total_model_size / len(all_model_runs)
                }

                #Both the configuration and the averages are saved together
                #To save to a .txt file.
                config_and_results = {
                    "config" : config,
                    "average" : total_average
                }
                
                #Save the config param and averages
                all_config_results.append(config_and_results)

                #Write into the .txt the results and the configurations parameters.
                file.write(f"Configuration {config_amount}\n")
                file.write(f"Parameters: {config_and_results['config']}\n")
                file.write(f"Averages: \n")
                
                for metric_name, value in config_and_results["average"].items():
                    file.write(f"   {metric_name}: {value:.6f}\n")
                               
                file.write("-" * 50 + "\n\n")
                config_amount += 1

                print(f"Finished run {i}.")
        print("Completed Model runs. See .txt file in results.")


"""
-------------------- Called HERE --------------------
"""
if __name__ == "__main__":
    print("Client started\n")
    client = ClientController()
    #mlp, 1d, xgboost, lstm
    client.client_code("svc", 5, svc_para)