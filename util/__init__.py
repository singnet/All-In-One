from dataset.aflw import AflwDataset
from dataset.celeba import CelebAAlignedDataset
from dataset.imdb_wiki import ImdbWikiDataset


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
    @staticmethod
    def from_string(string):
        return DatasetType.ds_types[string.lower()]
    
class Config(object):
    """Class that has information required to train model.
    """
    
    def __init__(self,dataset=None,dataset_dir=None,image_shape=None,epochs=None,batch_size=None,lr=None,steps_per_epoch=None,resume_model=False,large_model_name=None,small_model_name=None,model_weight=None,loss_weights=None):
        self.dataset = self.getDatasetFromString(dataset,dataset_dir,image_shape)
        self.epochs = epochs
        self.batch_size=batch_size
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.dataset_dir = None
        self.resume_model = resume_model
        self.large_model_name = large_model_name
        self.small_model_name = small_model_name
        self.model_weight = model_weight
        self.loss_weights = loss_weights
        self.image_shape = image_shape
        self.dataset_type = DatasetType.from_string(dataset)
    def getDatasetFromString(self, dataset_name,dataset_dir,image_shape):
        if dataset_name.lower() == "celeba_aligned":
            return CelebAAlignedDataset(dataset_dir,image_shape) 
        elif dataset_name.lower() == "wiki" or dataset_name.lower()=="imdb":
            return ImdbWikiDataset(dataset_dir,image_shape,dataset=dataset_name.lower())
        elif dataset_name.lower() == "aflw":
            return AflwDataset(dataset_dir,image_shape)
        else:
            raise NotImplementedError("Not implemented for "+str(dataset_name))

    def getDataset(self):
        return self.dataset
    def getEpochs(self):
        return self.epochs
    def getBatchSize(self):
        return self.batch_size
    def getLearingRate(self):
        return self.lr
    def getStepsPerEpoch(self):
        return self.steps_per_epoch
    