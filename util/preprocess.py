from scipy.io import loadmat
import os
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle

IMG_SIZE = (227,227)



def draw_rects(image,rects,bbox):
    print abs(bbox[0] - bbox[2]),"x",abs(bbox[1] - bbox[3])
    for rect in rects:  
        iou = bb_intersection_over_union(bbox,rect)
        if iou>=0.5:
            cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),2)
        elif abs(rect[0] - rect[2])>100 and abs(rect[1] - rect[3])>100 :
            cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
            
        # else:
        #     cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
            
    cv2.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
    return image

def rect_intersection(rect1,rect2):
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def rect_union(rect1,rect2):
    
    assert rect1.shape == (4,) , "rect1 shape should be (4,) and it is "+str(rect1.shape)
    assert rect2.shape == (4,) , "rect2 shape should be (4,) and it is "+str(rect2.shape)
    
    width1 = np.abs(rect1[0]-rect1[2])
    height1 = np.abs(rect1[1]-rect1[3])

    width2 = np.abs(rect2[0]-rect2[2])
    height2 = np.abs(rect2[1]-rect2[3])
    area1 = width1 * height1
    area2 = width2 * height2

    return area1+area2 - rect_intersection(rect1,rect2)


def bb_intersection_over_union(boxA, boxB):
    intr = rect_intersection(boxA,boxB)
    if(intr<=0):
        return 0
    runion = rect_union(boxA,boxB)
    if(runion<=0):
        return 0
    iou = intr / float(runion)
    return iou






class Preprocessor(object):
    def __init__(self):
        pass

    def load_dataset(self):
        raise Exception("Not implemented yet");
    def train_model(self,model):
        raise Exception("Not implemented yet");
    def generator(self):
        raise Exception("Not implemented yet");
class ImdbWikiDatasetPreprocessor(Preprocessor):
    def __init__(self,images_dir,dataset_type,load_db=False):
        super(ImdbWikiDatasetPreprocessor,self).__init__()
        self.dataset_type = dataset_type
        self.images_dir = images_dir
        self.dataset_loaded = False
        self.train_dataset,self.test_dataset = self.load_dataset()

    def get_meta(self,annotation_file_path):
        meta = pd.read_pickle(annotation_file_path)
        return meta
    def load_dataset(self):
        if self.images_dir is None:
            raise Exception("Images dir is None")
        elif not os.path.exists(self.images_dir):
            raise Exception("images dir path "+self.images_dir+" does not exist")
       
        train_dataset = self.get_meta(os.path.join(self.images_dir,"train.pkl"))
        test_dataset = self.get_meta(os.path.join(self.images_dir,"test.pkl"))
        train_dataset = self.remove_defected_data(train_dataset)
        test_dataset = self.remove_defected_data(test_dataset)
        test_dataset = test_dataset[:1000]
        test_dataset = self.load_images(test_dataset)
        self.dataset_loaded = True
        return train_dataset,test_dataset
    def calc_age(self,taken, dob):
        birth = datetime.fromordinal(max(int(dob) - 366, 1))
        if birth.month < 7:
            return taken - birth.year
        else:
            return taken - birth.year - 1
    def remove_defected_data(self,dataset):
        dataset = dataset[np.isfinite(dataset['score'])]
        dataset = dataset.reset_index(drop=True)
        dataset = dataset[np.isfinite(dataset['gender'])]
        dataset = dataset.reset_index(drop=True)
        dataset = dataset[np.isfinite(dataset['age'])]
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.loc[dataset["age"]>0]
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.loc[dataset["age"]<150]
        dataset = dataset.reset_index(drop=True)
        
        
        return dataset
    def load_images(self,dataset):
        # output = pd.DataFrame(columns=["image","age","gender"])
        output_images = np.zeros((len(dataset),227,227,3))
        output_ages = np.zeros((len(dataset)))
        output_genders = np.zeros((len(dataset)))
        for index,row in dataset.iterrows():
            img = cv2.imread(os.path.join(self.images_dir,row["file_name"][0]))
            if img is None:
                print row["file_name"]
                continue
            if row["score"]==float("-inf") or row["score"]==float("-inf"):
                cv2.imshow("No face annotated",img)
                cv2.waitKey(0)
                cv2.destroyAllWindows() 
            else:
                face_location = row["face_location"][0].astype(int) 
                face_image = img[face_location[1]:face_location[3],face_location[0]:face_location[2]]
                face_image = cv2.resize(face_image,IMG_SIZE).astype("float32")/255
                output_images[index] = face_image
                output_ages[index] = row["age"]
                output_genders[index] = row["gender"]

        output_genders = output_genders.astype(np.uint8)
        return output_images,output_ages,output_genders
    def generator(self,batch_size=32):
        while True:
            self.train_dataset = self.train_dataset.sample(frac=1).reset_index(drop=True)
            for i in range(0,len(self.train_dataset)-batch_size,batch_size):
                current_dataset = (self.train_dataset[i:i+batch_size]).reset_index(drop=True)
                X,age,gender = self.load_images(current_dataset)
                gender_out = np.eye(2)[gender]
                yield X,[age,gender_out]
class CelebADatasetPreprocessor(Preprocessor):
    def __init__(self):
        pass
class YaleDatasetPreprocessor(Preprocessor):
    def __init__(self):
        pass
class AlfwDatasetPreprocessor(Preprocessor):
    def __init__(self):
        pass