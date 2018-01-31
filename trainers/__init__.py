class Trainer(object):
    """ Class that abstracts training a given network on a given dataset.
    Parameters
    ----------
    preprocessor : preprocessors.BasicPreprocessor
        An instance of preprocessors.BasicPreprocessor or an instance of a class which inherits
        from preprocessors.BasicPreprocessor. Trainer object uses this instances to load
        and generate dataset. 
    network : nets.AllVsAllNetwork
        Network which have model attribute which will be trained using dataset given by preprocessor.
    """

    def __init__(self,preprocessor,network):
        self.preprocessor = preprocessor
        self.network = network
    def train(self):
        pass