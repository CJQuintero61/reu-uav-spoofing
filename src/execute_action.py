#Import abstract file
from drone_basics.abstracts import AbstractAction

class TrainAction(AbstractAction):
    def execute(self, model, data):
        print (f"Train Action on {model}")
        return model.train_model(data)

class TestAction(AbstractAction):
    def execute(self, model, data):
        print (f"Test Action on {model}")
        return model.predict(data)
    
class EvaluateAction(AbstractAction):
    def execute(self, model, data):
        print (f"Evaluate Action on {model}")
        return model.evaluate(data)