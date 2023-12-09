from abc import ABC, abstractmethod

class Trainer(ABC):

    @abstractmethod
    def train_epoch(self):
        '''
            The epoch is composed by 4 main steps:
            - Forward pass for all the batch in the training loader
            - Loss calculation using Loss callback
            - Gradient calculus and backward pass
            - Metric calculus and logging stuff
        '''
        return NotImplemented
    
    @abstractmethod
    def evaluate_epoch(self):
        '''
            The epoch is composed by 3 main steps:
            - Forward pass for all the batch in the validation loader
            - Loss calculation using Loss callback
            - Metric calculus and logging stuff
        '''
        return NotImplemented
    
    @abstractmethod
    def train(self):
        '''
            This method call the train and evaluation epoch step
            and set-up the training process
        '''
        return NotImplemented
    

