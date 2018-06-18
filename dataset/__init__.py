from abc import ABCMeta
from abc import abstractmethod
import os
from loggers import Log
import numpy as np
import pandas as pd

class Dataset(object):
    __metaclass__ = ABCMeta
    """Base class for all classes that abstract the datasets.
    Conventions
        - All dataset should have reference file location of images
        relative to dataset_dir variable.
        - All dataset should have all.pkl file contains dataframe that
        have  file_location key to load images
        - Dataset may contain train.pkl, test.pkl and validation.pkl
        inside dataset_dir. If they are not available the class that
        abstracts the dataset should create them when the object of
        that class is instantiated.
        N.B validation.pkl is optional.

    Parameters
    ----------
    dataset_dir : str
        Relative or absolute path to Dataset directory
    image_shape : numpy.ndarray
        Shape of image the dataset will be given to training network.
    Raises
    AssertionError
        if the dataset path does not exist.
    """

    def __init__(self,config):
        assert os.path.exists(config.dataset_dir),"Dataset directory: '"+config.dataset_dir+"' does not exist"
        self.dataset_loaded = False
        self.config = config
    """Loads train,test and validation datasets from files inside directory dataset_dir.
    load_dataset method should set dataset_loaded to True after sucessfully loading the dataset.
    The method should check if train.pkl and test.pkl files exists.
    Raises
    ------
    Exception
        if self.dataset_dir is None or self.dataset_dir path does not exists
    IOError
        if one or more of the files(train.pkl, test.pkl,validatin.pkl) do not exist
        inside self.dataset_dir directory, or if image file is missing for test or validation
        dataset.

    """
    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError("Not implmented!")
    @abstractmethod
    def generator(self,batch_size):
        raise NotImplementedError("Not implmented!")

    """Loads face images according to specification of dataset document.
    """

    @abstractmethod
    def load_images(self,dataframe):

        raise NotImplementedError("Not implemented")

    """This method checks the dataset directory has structure specified
    by this class's convention.
    """
    def has_met_convention(self):
        if not self.contain_dataset_files():
            return False
        else:
            if not self.dataset_loaded:
                self.load_dataset()
            if not "file_location" in self.train_dataset.columns:
                print "Failed to meet convention","training dataset does not contain file_location column"
                return False
            elif not "file_location" in self.test_dataset.columns:
                print "Failed to meet convention","test dataset does not contain file_location column"
                return False
            return True
    """ Loads dataset inside given pickle file.
    Parameters
    ----------
    ds_path : str
        Path to the pickle file
    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe of dataset with given file path.
    Raises
    ------
    IOError
        If the given file path doesnot exists
    """

    def get_meta(self,ds_path):
        dataframe = pd.read_pickle(ds_path)
        dataframe = dataframe.reset_index(drop=True)
        return dataframe
    """Checks if all.pkl, train.pkl and test.pkl files exists.
    Returns
    -------
    bool
        True if all the files exist. False if at least one of the files missing
    """

    def contain_dataset_files(self):
        if not os.path.exists(os.path.join(self.config.dataset_dir,"all.pkl")):
            print "Failed to meet convention","all.pkl is not inside:"+self.config.dataset_dir
            return False
        if not os.path.exists(os.path.join(self.config.dataset_dir,"train.pkl")):
            print "Failed to meet convention","train.pkl is not inside:"+self.config.dataset_dir
            return False
        if not os.path.exists(os.path.join(self.config.dataset_dir,"test.pkl")):
            print "Failed to meet convention","test.pkl is not inside:"+self.config.dataset_dir
            return False
        return True;

    @abstractmethod
    def meet_convention(self):
        raise NotImplementedError("Not implmented")
    def split_train_test(self,dataset,train_size=0.8):
        mask = np.random.rand(len(dataset)) < train_size
        train = dataset[mask]
        test = dataset[~mask]
        return train,test
    def split_train_test_validation(self,dataframe,train_size=0.8):
        train,test_val = self.split_train_test(dataframe,train_size)
        test_mask = np.random.rand(len(test_val))< 0.5
        test = test_val[test_mask]
        validation = test_val[~test_mask]
        return train,test,validation
    @abstractmethod
    def fix_labeling_issue(self,dataset):
        pass
    def get_column(self,dataframe,column):
        if dataframe is None:
            return None
        elif column in dataframe.columns:
            return dataframe[column].as_matrix()
        else:
            raise KeyError("dataframe does not contain column '"+label+"'")
    @abstractmethod
    def get_dataset_name(self):
        pass
