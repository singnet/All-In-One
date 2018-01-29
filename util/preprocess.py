from scipy.io import loadmat
import os
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle
import dlib

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
        self.train_dataset,self.test_dataset,_ = self.load_dataset()

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
        test_dataset = test_dataset[:100]
        # test_images = test_images[:100]
        test_images = self.load_images(test_dataset)
        self.dataset_loaded = True
        return train_dataset,[test_images,test_dataset],None
    
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

        return output_images
    def generator(self,batch_size=32):
        while True:
            self.train_dataset = self.train_dataset.sample(frac=1).reset_index(drop=True)
            for i in range(0,len(self.train_dataset)-batch_size,batch_size):
                current_dataset = (self.train_dataset[i:i+batch_size]).reset_index(drop=True)
                X = self.load_images(current_dataset)
                age = current_dataset["age"].as_matrix()
                gender = current_dataset["gender"].as_matrix().astype(np.uint8)
                gender_out = np.eye(2)[gender]
                yield X,[age,gender_out]
class CelebADatasetPreprocessor(Preprocessor):
    def __init__(self,dataset_dir,labels = ["Smiling"],aligned=True):
        self.labels = labels
        self.dataset_dir = dataset_dir
        self.dataset_type = "celeba"
        self.detector = dlib.get_frontal_face_detector()
        self.train_dataset, self.test_dataset,self.validation_dataset = self.load_train_test_dataset()
    def load_dataset_from_annotation_file(self):
        annotation_file = os.path.join(self.dataset_dir,"list_attr_celeba.txt")
        headers = ['imgfile', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        df = pd.read_csv(annotation_file,sep= "\s+|\t+|\s+\t+|\t+\s+",names=headers,header=1)
        return df
    def load_train_test_dataset(self):
        train = pd.read_pickle(os.path.join(self.dataset_dir,"train.pkl")).reset_index(drop=True)
        train = self.convertToOneAndZero(train)

        test = pd.read_pickle(os.path.join(self.dataset_dir,"test.pkl")).reset_index(drop=True)
        validation = pd.read_pickle(os.path.join(self.dataset_dir,"validation.pkl")).reset_index(drop=True)
        # test = test[:100].reset_index(drop=True)
        # validation = validation[:100].reset_index(drop=True)
        test_images = self.load_images(test)
        validation_images = self.load_images(validation)

        return train,[test_images,test],[validation_images,validation]
    def split_train_test(self,dataset,train_size=0.8):
        mask = np.random.rand(len(dataset)) < train_size
        train = dataset[mask]
        test_val = dataset[~mask]
        test_mask = np.random.rand(len(test_val))< 0.5
        test = test_val[test_mask]
        validation = test_val[~test_mask]
        return train,test,validation
    def load_images(self,dataset):
        
        output_images = np.zeros((len(dataset),227,227,3))
        for index,row in dataset.iterrows():
            img = cv2.imread(os.path.join(self.dataset_dir,row["imgfile"]))
            if img is None:
                print os.path.join(self.dataset_dir,row["imgfile"])
                continue
            faces = self.detector(img)
            if(len(faces)>0):
                face_location = faces[0]
                face_image = img[face_location.top():face_location.bottom(),face_location.left():face_location.right()]
                face_image = cv2.resize(face_image,IMG_SIZE).astype("float32")/255
                output_images[index] = face_image
            else:
                face_image = cv2.resize(img,IMG_SIZE).astype("float32")/255
                output_images[index] = face_image
                

        return output_images

    def convertToOneAndZero(self,dataset):
        for label in self.labels:
            dataset[label] = dataset[label]/2.0 + 1/2.0
        return dataset
    def generator(self,batch_size=32):
        while True:
            # shuffle train dataset
            self.train_dataset = self.train_dataset.sample(frac=1).reset_index(drop=True)
            for i in range(0,len(self.train_dataset)-batch_size,batch_size):
                current_dataset = (self.train_dataset[i:i+batch_size]).reset_index(drop=True)
                X = self.load_images(current_dataset)
                smile = current_dataset["Smiling"].as_matrix().astype(np.uint8)
                smile = np.eye(2)[smile]
                yield X,smile
    
    
    
class YaleDatasetPreprocessor(Preprocessor):
    def __init__(self):
        pass
class AlfwDatasetPreprocessor(Preprocessor):
    def __init__(self):
        pass

if __name__ == "__main__":
    preprocessor = CelebADatasetPreprocessor("/home/mtk/datasets/img_align_celeba")
    train,test,val = preprocessor.load_train_test()
    print train[:10]["Smiling"]
    train["Smiling"] = train["Smiling"]/2.0 +1/2.0 
    test["Smiling"] = test["Smiling"]/2.0 +1/2.0 
    val["Smiling"] = val["Smiling"]/2.0 +1/2.0

    