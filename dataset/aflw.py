from dataset import Dataset
import sqlite3
import dlib
import cv2

class AflwDataset(Dataset):
    """Class that abstracts Aflw dataset.
    """

    def __init__(self,dataset_dir,image_shape):
        self.conn = sqlite3.connect("/home/mtk/dataset/aflw-files/aflw/data/aflw.sqlite")
    
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

    def selective_search(self,img,min_size=(2200),max_img_size=(500,500),debug=False):
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
        raise NotImplementedError("Not implmented!")
    def generator(self,batch_size):
        raise NotImplementedError("Not implmented!")
    def load_images(self,dataframe):
        raise NotImplementedError("Not implmented!")
    def meet_convention(self):
        raise NotImplementedError("Not implmented!")
    def fix_labeling_issue(self,dataset):
        raise NotImplementedError("Not implmented!")
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
        pass
    def iou(self,rect):
        pass
    def __str__(self):
        return "("+str(self.x)+","+self.y+") (" +str(self.w)+","+self.h+")"