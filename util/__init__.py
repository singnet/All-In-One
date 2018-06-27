from dataset.aflw import AflwDataset
from dataset.celeba import CelebAAlignedDataset
from dataset.imdb_wiki import ImdbWikiDataset
from dataset.adience import AdienceDataset
import argparse

class DatasetType(object):


    ds_types = {
        "imdb":0,   # gender and age dataset
        "wiki":1,   # gender and age dataset
        "celeba":2, # gender,identity and smile dataset
        "celeb_a":2,# gender,identity and smile dataset
        "yale":3,   # pose,identity and illumunation dataset
        "ck+":4,    # emotion,identity and dataset
        "aflw":5,   # pose, key points,detection
        "test":6,
        "adience":6,   # dataset used solely for testing the methods of classes. It is found inside tests/ds/ folder.
    }
    IMDB = 0
    WIKI = 1
    CELEB_A = 2
    YALE = 3
    CK_PLUS = 4
    ADIENCE = 5

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

    def __init__(self,dataset,dataset_dir,image_shape,epochs=None,batch_size=None,lr=None,steps_per_epoch=None,resume_model=False,large_model_name=None,small_model_name=None,model_weight=None,loss_weights=None):
        self.epochs = epochs
        self.batch_size=batch_size
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.dataset_dir = dataset_dir
        self.resume_model = resume_model
        if large_model_name == None or large_model_name.strip()=="":
            self.large_model_name = dataset + "-large-model"
        else:
            self.large_model_name = large_model_name
        if small_model_name == None or small_model_name.strip() == "":
            self.small_model_name = dataset + " -small-model"
        else:
            self.small_model_name = small_model_name
        self.model_weight = model_weight
        if loss_weights is None:
            self.loss_weights = {}
        else:
            self.loss_weights = loss_weights
        self.image_shape = image_shape
        self.dataset_type = DatasetType.from_string(dataset)
        self.dataset = dataset
    def getEpochs(self):
        return self.epochs
    def getBatchSize(self):
        return self.batch_size
    def getLearningRate(self):
        return self.lr
    def getStepsPerEpoch(self):
        return self.steps_per_epoch



def get_cmd_args():
    parser = argparse.ArgumentParser()
    # --mtype is model type argument. it can be either 'np'(neutral vs positive emotion classifier) or 'ava'(All basic
    # seven emotion classifier[anger,fear,disgust,happy,sad,surprise,neutral]). Default is 'np'
    parser.add_argument("--images_path",default="",type=str)
    parser.add_argument("--dataset",default="",type=str)
    parser.add_argument("--label",default="",type=str)
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    parser.add_argument("--lr",default=1e-4,type=float)
    parser.add_argument("--load_db",default=False,type=bool)
    parser.add_argument("--resume",default = False,type=bool)
    parser.add_argument("--steps",default = 100,type=int)
    parser.add_argument("--ol",default = "large_model",type=str)
    parser.add_argument("--os",default = "",type=str)
    parser.add_argument("--load_model",default=None,type=str)
    parser.add_argument("--age_loss_weight",default=0.5,type=float)
    parser.add_argument("--gender_loss_weight",default=1,type=float)
    parser.add_argument("--smile_loss_weight",default=0.5,type=float)
    parser.add_argument("--detection_loss_weight",default=1,type=float)
    parser.add_argument("--visibility_loss_weight",default=1,type=float)
    parser.add_argument("--pose_loss_weight",default=1,type=float)
    parser.add_argument("--landmarks_loss_weight",default=1,type=float)
    parser.add_argument("--identity_loss_weight",default=1,type=float)
    parser.add_argument("--eye_glasses_loss_weight",default=1,type=float)
    parser.add_argument("--freeze",default=False,type=bool)

    args = parser.parse_args()
    return args


def get_config(args):
    config = Config(args.dataset,args.images_path,(227,227,1))
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    config.steps_per_epoch = args.steps
    config.model_weight = args.load_model
    config.resume = args.resume
    config.large_model_name = args.ol
    config.small_model_name = args.os
    config.loss_weights = {
            "age": args.age_loss_weight,
            "gender" : args.gender_loss_weight,
            "detection": args.detection_loss_weight,
            "visiblity": args.visibility_loss_weight,
            "pose": args.pose_loss_weight,
            "landmarks": args.landmarks_loss_weight,
            "identity": args.identity_loss_weight,
            "smile": args.smile_loss_weight,
            "eye_glasses": args.eye_glasses_loss_weight
    }
    config.label = args.label
    return config
