from __future__ import print_function
from dataset import Dataset
import os
from inspect import currentframe, getframeinfo
from loggers import Log
import pandas as pd
import cv2
import dlib
import numpy as np

class CelebAAlignedDataset(Dataset):
    """Class that abstracts celeba aligned dataset.
    Assumtions
        - Dataset contains images inside dataset_dir
        - list_attr_celeba.txt inside dataset_dir
        - list_landmarks_align_celeba.txt inside dataset_dir
    Parameters
    ----------
    dataset_dir : str
        Relative or absolute path to dataset folder
    Raises
    AssertionError
        if the dataset path does not exist.
    """

    def __init__(self,config,labels=["Smiling","Male"]):
        super(CelebAAlignedDataset,self).__init__(config)
        self.labels = labels
        self.detector = dlib.get_frontal_face_detector()

    def load_dataset(self):
        if set(self.labels).issubset(["Smiling","Male"]):
            if not self.contain_dataset_files():
                self.meet_convention()
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading pickle files")
            Log.DEBUG_OUT =False
            self.train_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"train.pkl"))
            self.test_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"test.pkl"))
            if os.path.exists(os.path.join(self.config.dataset_dir,"validation.pkl")):
                self.validation_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"validation.pkl"))
            else:
                self.validation_dataset = None
                frameinfo = getframeinfo(currentframe())
                Log.WARNING("Unable to find validation dataset",file_name=__name__,line_number=frameinfo.lineno)

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
                img = cv2.imread(os.path.join(self.config.dataset_dir,row["file_location"]))

                if img is None:
                    Log.WARNING("Unable to read images from "+os.path.join(self.config.dataset_dir,row["file_location"]))
                    continue
                faces = self.detector(img)
                for i in range(len(faces)):
                    if(len(faces)>0):
                        face_location = faces[i]
                        face_image = img[face_location.top():face_location.bottom(),face_location.left():face_location.right()]
                        try:
                            face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
                            face_image = cv2.resize(face_image,(self.config.image_shape[0],self.config.image_shape[1]))
                            output_images[index] = face_image.reshape(self.config.image_shape)
                            break
                        except:
                            # Log.ERROR_OUT = True
                            # Log.ERROR ("error"+","+str(face_image is None)+","+str(img is None)+","+ str(len(faces)))
                            # Log.ERROR (str(face_location.top())+","+ str(face_location.bottom())+","+str(face_location.left())+","+str(face_location.right()))
                            print (str(face_location.top())+","+ str(face_location.bottom())+","+str(face_location.left())+","+str(face_location.right()))
                            print ("error"+","+str(face_image is None)+","+str(img is None)+","+ str(len(faces)))
                    else:
                        face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
                        face_image = cv2.resize(img,(self.config.image_shape[0],self.config.image_shape[1]))
                        output_images[index] = face_image.reshape(self.config.image_shape)
                        Log.WARNING("Dlib unable to find faces from :"+os.path.join(self.config.dataset_dir,row["file_location"])+" Loading full image as face")
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
            dataframe = self.load_dataset_from_annotation_file()
            train,test,validation = self.split_train_test_validation(dataframe)
            train.to_pickle(os.path.join(self.config.dataset_dir,"train.pkl"))
            test.to_pickle(os.path.join(self.config.dataset_dir,"test.pkl"))
            validation.to_pickle(os.path.join(self.config.dataset_dir,"validation.pkl"))
            dataframe.to_pickle(os.path.join(self.config.dataset_dir,"all.pkl"))

    def load_dataset_from_annotation_file(self):
        annotation_file = os.path.join(self.config.dataset_dir,"list_attr_celeba.txt")
        headers = ['file_location', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        df = pd.read_csv(annotation_file,sep= "\s+|\t+|\s+\t+|\t+\s+",names=headers,header=1)
        return df
    def generator(self,batch_size):
	raise NotImplementedError("Not implmented")
    def smile_data_generator(self,batch_size=32):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                X = X.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
                smile = self.get_column(current_dataframe,"Smiling").astype(np.uint8)
                smile = np.eye(2)[smile]
                yield X,smile

    def fix_labeling_issue(self,dataset):
        if dataset is None:
            return None
        output = dataset.copy(deep=True)
        for label in self.labels:
            output[label] = output[label]/2.0 + 1/2.0
        return output
    def get_dataset_name(self):
        return "celeba"
