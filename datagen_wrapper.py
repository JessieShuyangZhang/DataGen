from abc import ABC

class DataGenWrapper(ABC):
    '''
    abstract base class: generic data-generation wrapper
    '''
    def fit(self, *args, **kwargs): # train the model
        pass

    def generate(self, *args, **kwargs): # return synthetic data
        pass

    # not implemented in child class yet:

    # def save_model(self, filename):
    #     pass

    # def load_model(self, filename):
    #     pass
    