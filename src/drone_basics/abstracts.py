#Abstract factory
from abc import ABC, abstractmethod

#Model selection module
class AbstractModel(ABC):
    #Selects the model type and do that models necessary setup of the factory
    @abstractmethod
    def train_model(self, data):
        pass
    
    @abstractmethod
    def predict(self, data):
        pass
    
    @abstractmethod
    def evaluate(self, data):
        pass

#Action selection
class AbstractAction(ABC):
    @abstractmethod
    def execute(self, model, data):
        pass