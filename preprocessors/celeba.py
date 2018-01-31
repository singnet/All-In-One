from preprocessors import AbastractPreprocessor
import numpy as np
import os
import cv2
import dlib
import pandas as pd

class CelebAPreprocessor(AbastractPreprocessor):
    """Class that preprocesses CelebA dataset.
    for more information visit preprocessors.AbastractPreprocessor
    """

    def __init__(self,dataset_type,dataset_dir,split=False,face_image_shape=(227,227,3),aligned=True,labels=["Smiling"]):
        super(CelebAPreprocessor,self).__init__(dataset_type,dataset_dir,split=split,face_image_shape=face_image_shape)
        self.detector = dlib.get_frontal_face_detector()
        self.aligned = aligned
        self.labels = labels
    
    def load_faces(self,dataset):
        if self.aligned:
            output_images = np.zeros((len(dataset),self.face_image_shape[0],self.face_image_shape[1],self.face_image_shape[2]))
            for index,row in dataset.iterrows():
                img = cv2.imread(os.path.join(self.dataset_dir,row["imgfile"]))
                if img is None:
                    print os.path.join(self.dataset_dir,row["imgfile"])
                    continue
                faces = self.detector(img)
                if(len(faces)>0):
                    face_location = faces[0]
                    face_image = img[face_location.top():face_location.bottom(),face_location.left():face_location.right()]
                    face_image = cv2.resize(face_image,(self.face_image_shape[0],self.face_image_shape[1])).astype("float32")/255
                    output_images[index] = face_image
                else:
                    face_image = cv2.resize(img,(self.face_image_shape[0],self.face_image_shape[1])).astype("float32")/255
                    output_images[index] = face_image
            return output_images
        else:
            raise NotImplementedError("load_faces method is not implemented for non aligned celebA dataset")
    def fix_labeling_issue(self,dataset):
        output = dataset.copy(deep=True)
        for label in self.labels:
            output[label] = output[label]/2.0 + 1/2.0
        return output
    def generator(self,batch_size=32):
        pass