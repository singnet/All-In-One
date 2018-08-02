from dataset import Dataset
import sqlite3
import dlib
import cv2
import os
import numpy as np
import pandas as pd
from loggers import Log


class AflwDataset(Dataset):
    """Class that abstracts Aflw dataset.
    """

    def __init__(self,config):
        super(AflwDataset,self).__init__(config)

    """ method that resizes image to the same resolution
    image which have width and height equal or less than
    values specified by max_size.
        e.g
        img = np.zeros((200,300))
        img = resize_down_image(img,(100,100))
        img.shape # (66,100)

    """

    def resize_down_image(self,img,max_img_shape):
        img_h,img_w = img.shape[0:2]
        w, h = img_w,img_h
        if max_img_shape[0]<h:
            w = (max_img_shape[0]/float(h))  * w
            h = max_img_shape[0]
        if max_img_shape[1]<w:
            h = (max_img_shape[1]/float(w)) * h
            w = max_img_shape[1]
        if h == img_h:
            return img,1
        else:
            scale = img_h/h
            img = cv2.resize(img,(int(w),int(h)))
            return img,scale

    def selective_search(self,img,min_size=(2200),max_img_size=(24,24),debug=False):
        cand_rects = []
        img,scale = self.resize_down_image(img,max_img_size)
        dlib.find_candidate_object_locations(img,cand_rects,min_size=min_size)
        rects = [(int(crect.left() * scale),
             int(crect.top()* scale),
             int(crect.right()* scale),
             int(crect.bottom()* scale),

            ) for crect in cand_rects]
        for rect in rects:
            cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_dataset(self):
        if self.config.label == "detection":
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
            test_indexes = np.arange(len(self.test_dataset))
            np.random.shuffle(test_indexes)
            validation_indexes = np.arange(len(self.validation_dataset))
            np.random.shuffle(validation_indexes)

            self.test_dataset = self.test_dataset.iloc[test_indexes].reset_index(drop=True)
            self.validation_dataset = self.validation_dataset.iloc[validation_indexes].reset_index(drop=True)

            self.test_dataset = self.test_dataset[:1000]
            self.validation_dataset = self.validation_dataset[:100]
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading test images")
            Log.DEBUG_OUT =False
            self.test_dataset_images = self.load_images(self.test_dataset).astype(np.float32)/255
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading validation images")
            Log.DEBUG_OUT =False
            self.validation_dataset_images = self.load_images(self.validation_dataset).astype(np.float32)/255
            self.test_detection = self.test_dataset["is_face"].as_matrix()
            self.dataset_loaded = True
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded all dataset and images")
            Log.DEBUG_OUT =False

        else:
            raise NotImplementedError("Not implemented for labels:"+str(self.labels))

    def generator(self,batch_size):
        raise NotImplementedError("Not implmented!")
        
    def detection_data_genenerator(self,batch_size):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                X = X.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
                detection = self.get_column(current_dataframe,"is_face").astype(np.uint8)
                detection = np.eye(2)[detection]
                yield X,detection

    def load_images(self,dataframe):
        output_images = np.zeros((len(dataframe),self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2]))
        for index,row in dataframe.iterrows():
            file_location = row["file_location"]
            img = cv2.imread(file_location)
            if img is None:
                print("Unable to read image from ",file_location)
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = scipy.misc.imresize(img, (self.config.image_shape[0], self.config.image_shape[1])).astype(np.float32)/255
            output_images[index] = img.reshape(self.config.image_shape)
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
            dataframe = self.load_face_non_face_dataset()
            train,test,validation = self.split_train_test_validation(dataframe)
            train.to_pickle(os.path.join(self.config.dataset_dir,"train.pkl"))
            test.to_pickle(os.path.join(self.config.dataset_dir,"test.pkl"))
            validation.to_pickle(os.path.join(self.config.dataset_dir,"validation.pkl"))
            dataframe.to_pickle(os.path.join(self.config.dataset_dir,"all.pkl"))

    def load_face_non_face_dataset(self):
        output_file_locations = []
        output_is_face = []
        for img_path in os.listdir(os.path.join(self.config.dataset_dir,"face")):
            output_file_locations+=[os.path.join(self.config.dataset_dir,"face",img_path)]
            output_is_face+=[1]
        for img_path in os.listdir(os.path.join(self.config.dataset_dir,"non-face")):
            output_file_locations+=[os.path.join(self.config.dataset_dir,"non-face",img_path)]
            output_is_face+=[0]
        output_df = pd.DataFrame(columns=["file_location","is_face"])
        output_df["file_location"] = output_file_locations
        output_df["is_face"] = output_is_face
        return output_df

    def fix_labeling_issue(self,dataset):
        return dataset

    def rect_intersection(self,rect1,rect2):
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea

    def rect_union(self,rect1,rect2):

        assert rect1.shape == (4,) , "rect1 shape should be (4,) and it is "+str(rect1.shape)
        assert rect2.shape == (4,) , "rect2 shape should be (4,) and it is "+str(rect2.shape)

        width1 = np.abs(rect1[0]-rect1[2])
        height1 = np.abs(rect1[1]-rect1[3])

        width2 = np.abs(rect2[0]-rect2[2])
        height2 = np.abs(rect2[1]-rect2[3])
        area1 = width1 * height1
        area2 = width2 * height2

        return area1+area2 - self.rect_intersection(rect1,rect2)

    def bb_intersection_over_union(self,boxA, boxB):
        intr = self.rect_intersection(boxA,boxB)
        if(intr<=0):
            return 0
        runion = rect_union(boxA,boxB)
        if(runion<=0):
            return 0
        iou = intr / float(runion)
        return iou

    def get_dataset_name(self):
        return "aflw"

class Rect(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def area(self):
        return self.w * self.h
    def intersection(self,rect):
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(self.x, rect.x));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea

    def union(self,rect):
        assert rect1.shape == (4,) , "rect1 shape should be (4,) and it is "+str(rect1.shape)
        assert rect2.shape == (4,) , "rect2 shape should be (4,) and it is "+str(rect2.shape)

        width1 = np.abs(rect1[0]-rect1[2])
        height1 = np.abs(rect1[1]-rect1[3])

        width2 = np.abs(rect2[0]-rect2[2])
        height2 = np.abs(rect2[1]-rect2[3])
        area1 = width1 * height1
        area2 = width2 * height2

        return area1+area2 - self.rect_intersection(rect1,rect2)
    def iou(self,rect):
        pass
    def __str__(self):
        return "("+str(self.x)+","+self.y+") (" +str(self.w)+","+self.h+")"
