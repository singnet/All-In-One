from __future__ import print_function
from dataset import Dataset
import os
from inspect import currentframe, getframeinfo
from loggers import Log
import pandas as pd
import cv2
import dlib
import numpy as np
from scipy.io import loadmat
from datetime import datetime

class ImdbWikiDataset(Dataset):
    """Class that abstracs Imdb and wiki datasets.
    Assumtions
        - Dataset contains images inside dataset_dir
        - either wiki.mat or imdb_crop.mat inside dataset_dir
    Parameters
    ----------
    dataset_dir : str
        Relative or absolute path to dataset folder
    Raises
    AssertionError
        if the dataset path does not exist.
    """

    def __init__(self,config,labels=["Age","Gender"]):
        super(ImdbWikiDataset,self).__init__(config)
        self.labels = labels
        self.dataset=config.dataset
    def load_dataset(self):
        if set(self.labels).issubset(["Age","Gender"]):
            if not self.contain_dataset_files():
                self.meet_convention()
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading pickle files")
            Log.DEBUG_OUT = False
            self.train_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"train.pkl"))
            self.test_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"test.pkl"))
            if os.path.exists(os.path.join(self.config.dataset_dir,"validation.pkl")):
                self.validation_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"validation.pkl"))
            else:
                self.validation_dataset = None
                frameinfo = getframeinfo(currentframe())
                Log.WARINING_OUT = True
                Log.WARNING("Unable to find validation dataset",file_name=__name__,line_number=frameinfo.lineno)
                Log.WARINING_OUT = False
            self.train_dataset = self.fix_labeling_issue(self.train_dataset)
            self.test_dataset = self.fix_labeling_issue(self.test_dataset)
            self.validation_dataset = self.fix_labeling_issue(self.validation_dataset)
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded train, test and validation dataset")
            Log.DEBUG_OUT =False
            self.test_dataset = self.test_dataset[:5000]
            self.validation_dataset = self.validation_dataset[:100]
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading test images")
            Log.DEBUG_OUT =False
            self.test_dataset_images = self.load_images(self.test_dataset).astype(np.float32)/255
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading validation images")
            Log.DEBUG_OUT =False
            self.validation_dataset_images = self.load_images(self.validation_dataset).astype(np.float32)/255
            self.dataset_loaded = True
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded all dataset and images")
            Log.DEBUG_OUT =False

        else:
            raise NotImplementedError("Not implemented for labels:"+str(self.labels))

    def load_images(self,dataframe):
        if dataframe is None:
            return None
        else:
            assert type(dataframe) == pd.core.frame.DataFrame, "argument to load image should be dataframe"
            assert  "file_location" in dataframe.columns, "dataframe should contain file_location column"
            output_images = np.zeros((len(dataframe),self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2]))
            for index,row in dataframe.iterrows():
                img = cv2.imread(os.path.join(self.config.dataset_dir,row["file_location"][0]))

                if img is None:
                    Log.WARNING("Unable to read images from "+os.path.join(self.config.dataset_dir,row["file_location"][0]))
                    continue
                face_location = row["face_location"][0].astype(int)
                face_image = img[face_location[1]:face_location[3],face_location[0]:face_location[2]]
                face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image,(self.config.image_shape[0],self.config.image_shape[1]))
                output_images[index] = face_image.reshape(self.config.image_shape)
            return output_images

    def meet_convention(self):
        if self.contain_dataset_files():
            return
        elif os.path.exists(os.path.join(self.config.dataset_dir,"all.pkl")):
            dataframe = pd.read_pickle(os.path.join(self.config.dataset_dir,"all.pkl"))
            train,test,validation = self.split_train_test_validation(dataframe)
            train.to_pickle(os.path.join(self.config.dataset_dir,"train.pkl"))
            test.to_pickle(os.path.join(self.config.dataset_dir,"test.pkl"))
            validation.to_pickle(os.path.join(self.config.dataset_dir,"validation.pkl"))
        else:
            dataframe = self.load_from_mat(os.path.join(self.config.dataset_dir,self.dataset+".mat"))
            train,test,validation = self.split_train_test_validation(dataframe)
            train.to_pickle(os.path.join(self.config.dataset_dir,"train.pkl"))
            test.to_pickle(os.path.join(self.config.dataset_dir,"test.pkl"))
            validation.to_pickle(os.path.join(self.config.dataset_dir,"validation.pkl"))
            dataframe.to_pickle(os.path.join(self.config.dataset_dir,"all.pkl"))

    def generator(self,batch_size=32):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                X = X.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
                age = self.get_column(current_dataframe,"Age")
                gender = self.get_column(current_dataframe,"Gender").astype(np.uint8)
                gender = np.eye(2)[gender]
                yield X,[age,gender]

    def age_data_genenerator(self,batch_size):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                age = self.get_column(current_dataframe,"Age")

                yield X,age

    def gender_data_genenerator(self,batch_size):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                gender = self.get_column(current_dataframe,"Gender").astype(np.uint8)
                gender = np.eye(2)[gender]
                yield X,gender


    def fix_labeling_issue(self,dataset):
        if dataset is None:
            return None
        output = dataset.copy(deep=True)

        output = output[np.isfinite(output['score'])]
        output = output.reset_index(drop=True)
        output = output[np.isfinite(output['Gender'])]
        output = output.reset_index(drop=True)
        output = output[np.isfinite(output['Age'])]
        output = output.reset_index(drop=True)
        output = output.loc[output["Age"]>0]
        output = output.reset_index(drop=True)
        output = output.loc[output["Age"]<150]
        output = output.reset_index(drop=True)

        return output

    def calc_age(self,taken, dob):
        birth = datetime.fromordinal(max(int(dob) - 366, 1))
        # assume the photo was taken in the middle of the year
        if birth.month < 7:
            return taken - birth.year
        else:
            return taken - birth.year - 1

    def load_from_mat(self,mat_path):
        meta = loadmat(mat_path)
        file_location = meta[self.dataset][0, 0]["full_path"][0]
        dob = meta[self.dataset][0, 0]["dob"][0]  # Matlab serial date number
        gender = meta[self.dataset][0, 0]["gender"][0]
        photo_taken = meta[self.dataset][0, 0]["photo_taken"][0]  # year
        face_score = meta[self.dataset][0, 0]["face_score"][0]
        second_face_score = meta[self.dataset][0, 0]["second_face_score"][0]
        age = [self.calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        face_location = meta[self.dataset][0, 0]["face_location"][0]

        df = pd.DataFrame.from_dict({"file_location":file_location,"Gender":gender,

        "score":face_score,"second_face_score":second_face_score,"Age":age,"face_location":face_location})

        return df
    def get_dataset_name(self):
        return self.dataset
