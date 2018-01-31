class DatasetType(object):
    
    
    ds_types = {
        "imdb":0,   # gender and age dataset
        "wiki":1,   # gender and age dataset
        "celeba":2, # gender,identity and smile dataset
        "celeb_a":2,# gender,identity and smile dataset
        "yale":3,   # pose,identity and illumunation dataset
        "ck+":4,    # emotion,identity and dataset
        "test":5    # dataset used solely for testing the methods of classes. It is found inside tests/ds/ folder.
    }
    IMDB = 0
    WIKI = 1
    CELEB_A = 2
    YALE = 3
    CK_PLUS = 4

    """ Class that abstracts all dataset type currently available.
    Using this class assumes all dataset available are listed above.
    Parameters
    ----------
    dataset_name : str
        Name of dataset that this class represent.abs
    """
    def __init__(self,dataset_name):
        self.dataset_type = self.from_string(dataset_name)
    """ convert dataset name given to integer representation of the dataset.
    Parameters
    ----------
    string : str
        name of the dataset
    Return
    ------
    int 
        integer that represents a dataset with given name
    """

    def from_string(self,string):
        return DatasetType.ds_types[string.lower()]
    