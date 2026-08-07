#Import abstract file
from abstracts import AbstractAction

class TrainAction(AbstractAction):
    def execute(self, model, data):
        print (f"Train Action called.")
        return model.train_model(data)

class TestAction(AbstractAction):
    def execute(self, model, data):
        print (f"Test Action called.")
        return model.predict(data)
    
class EvaluateAction(AbstractAction):
    def execute(self, model, data):
        print (f"Evaluate Action called.")
        return model.evaluate(data)