from util import DatasetType
import abc
import numpy as np
import pandas as pd
import cv2
import os
from abc import ABCMeta

class AbastractPreprocessor(object):
    __metaclass__ = ABCMeta
    """
    Parent class for all classes that abstract preprocessing of dataset.
    
    Parameters
    ----------
    dataset_type : str
        String name of dataset type. for more information please vist doc string of util.DatasetType class
    split: bool
        Split and save dataset inside all.pkl into train.pkl,test.pkl and validation.pkl
    face_image_shape : tuple
        Shape of face image to generate
    dataset_dir : str
        Absolute or relative path to the dataset folder. This class assumes the following file structure,
            ├──dataset_dir/
            ├─────── train.pkl
            ├─────── test.pkl
            ├─────── validation.pkl
            ├─────── all.pkl 
            ├─────── img1.png 
            └─────── img2.png
                       ...
                       ...
    """

    
    def __init__(self,dataset_type,dataset_dir,split=False,face_image_shape=(227,227,3)):
        self.dataset_type = DatasetType(dataset_type)
        self.dataset_dir = dataset_dir
        self.dataset_loaded = False
        self.face_image_shape = face_image_shape
        if(split):
            self.split_train_test_validation()

    """
    Method that loads images of a given dataset. The given dataset dataframe should contain file_location key.

    
    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataframe object that contains file location of images to be loaded

    Return 
    ------
    numpy.ndarray
        Array of images with respective order of file location in given dataset
    Raises
    ------
    KeyError
        If file_location key does not exist in dataset dataframe
    IOError
        If Io error occured while reading images
    """

    @abc.abstractmethod
    def load_faces(self,dataset):
        pass

    """ Loads datasets from files inside directory dataset_dir.
    Raises
    ------
    Exception
        if self.dataset_dir is None or self.dataset_dir path does not exists
    IOError
        if one or more of the files(train.pkl, test.pkl,validatin.pkl) do not exist
        inside self.dataset_dir directory, or if image file is missing for test or validation
        dataset.

    """

    def load_datasets(self):
        if self.dataset_dir is None:
            raise Exception("Images dir is None")
        elif not os.path.exists(self.dataset_dir):
            raise Exception("images dir path "+self.dataset_dir+" does not exist")
       
        train_dataset = self.get_meta(os.path.join(self.dataset_dir,"train.pkl"))
        test_dataset = self.get_meta(os.path.join(self.dataset_dir,"test.pkl"))
        validation_dataset = self.get_meta(os.path.join(self.dataset_dir,"validation.pkl"))

        # Remove datasets that have labeling problem
        self.train_dataset = self.fix_labeling_issue(train_dataset)
        self.test_dataset = self.fix_labeling_issue(test_dataset)
        self.validation_dataset = self.fix_labeling_issue(validation_dataset)

        self.test_images = self.load_faces(self.test_dataset)
        self.validation_images = self.load_faces(self.validation_dataset)
        self.dataset_loaded = True
    
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
        return pd.read_pickle(ds_path)
    

    """Fixes labeling issues such as missing images, missing label, incorrect labeling
    specified by the dataset document. This method also make labeling agree between 
    all dataset. e.g some dataset use for negetive attributes -1  while others use 0. 
    This method converts all -1 to 0. In this project 1 represents presence of(positive) attribute,
    and 0 absence of(negetive) attributes. The method should not modify the dataset passed as argument.
    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset inthe form of dataframe
    Returns
    -------
    pands.core.frame.DataFrame
        returns dataframe where the labeling issues is fixed
    """
    @abc.abstractmethod
    def fix_labeling_issue(self,dataset):
        pass

    """Splits dataset into train, test and validation. test and validation size are equal
    
    Parameters[summary]
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset to be splitted
    train_size : float
        Percentage of training dataset
    Returns
    ------
    tupple
        (train,test,validation)
    """

    def split_train_test_validation(self,train_size=0.8):
        dataset = self.get_meta(os.path.join(self.dataset_dir,"all.pkl"))
        mask1 = np.random.rand(len(dataset)) < train_size

        train_dataset = dataset[mask1].reset_index(drop=True)
        test_val_dataset = dataset[~mask1].reset_index(drop=True)

        mask2  = np.random.rand(len(test_val_dataset))< 0.5 


        test_dataset = test_val_dataset[mask2]
        validation_dataset = test_val_dataset[~mask2]

        train_dataset.to_pickle(os.path.join(self.dataset_dir,"train.pkl"))
        test_dataset.to_pickle(os.path.join(self.dataset_dir,"test.pkl"))
        validation_dataset.to_pickle(os.path.join(self.dataset_dir,"validation.pkl"))

        return len(train_dataset),len(test_dataset),len(validation_dataset)
    @abc.abstractmethod
    def generator(self,batch_size=32):
        pass
    """
    """

    @abc.abstractmethod
    def to_common_datasetform(self,dataset):
        pass
    