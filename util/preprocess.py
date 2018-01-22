from scipy.io import loadmat
import os
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import selectivesearch
from util.process_imdb_wiki import load_dataset
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

def search_face(img):
    pass
def getCurrentAgeGenderData(matrix,dataset_dir):
    output = pd.DataFrame(columns=["image","age","gender"])
    for index in range(len(matrix)):
        if  matrix[index][1]==float("-inf") or matrix[index][1]==float("-inf"):
            continue
        img = cv2.imread(os.path.join(dataset_dir,matrix[index][0][0]))
        if img is None:
            continue    
        face_location = matrix[index][2][0].astype(int) 
        face_image = img[face_location[1]:face_location[3],face_location[0]:face_location[2]]
        face_image = cv2.resize(face_image,IMG_SIZE)
        output.loc[-1] = [face_image, matrix[index][3], matrix[index][4]]  # adding a row
        output.index = output.index + 1  # shifting index
        output = output.sort_index()  
    return output
def age_gender_dataset_generator(dataset_matrix,dataset_dir,batch_size=32):
    print dataset_matrix.shape
    while True:
        indexes = range(dataset_matrix.shape[0])
        np.random.shuffle(indexes)
        for i in range(0,len(indexes)-batch_size,batch_size):
            current_indexes = indexes[i:i+batch_size]
            currentDataframe = dataset_matrix[current_indexes]
            curentDataset = getCurrentAgeGenderData(currentDataframe,dataset_dir)
            X = curentDataset["image"].values
            age = curentDataset["age"].values
            gender = curentDataset["gender"].values
            X_out = np.zeros((len(X),227,227,3))
            for i in range(X.shape[0]):
                X_out[i] = X[i]
            gender = gender.astype(np.uint8)
            gender = np.eye(2)[gender]
            age = age.reshape(-1,1)
            yield X_out,[age,gender]


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

def getAgeGenderDatasetHelper(dataset_dir,dataset_type):
    
    dataset = load_dataset(dataset_dir,dataset_type,["file_name","score","face_location","age","gender"])
    dataset = dataset.loc[dataset["score"]!=float("-inf")]
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.loc[dataset["gender"]!=float("nan")]
    dataset = dataset.reset_index(drop=True)
    output = pd.DataFrame(columns=["image","age","gender"])



    for index,row in dataset.iterrows():
        img = cv2.imread(os.path.join(dataset_dir,row["file_name"][0]))
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
            face_image = cv2.resize(face_image,IMG_SIZE)
            output.loc[-1] = [face_image, row["age"], row["gender"]]  # adding a row
            output.index = output.index + 1  # shifting index
            output = output.sort_index()  
        # if index>100:
        #     break
    return output

def getAgeGenderDataset(wiki_dir,imdb_dir):
    print "loading wiki"
    wiki_dataset = getAgeGenderDatasetHelper(wiki_dir,"wiki")
    # imdb_dataset = getAgeGenderDatasetHelper(imdb_dir,"imdb")
    print len(wiki_dataset)
    
   
    return wiki_dataset


def selectiveSearchOpencv(img):
    pass
# def test_dataset(di)

def test_age_of_wiki_dataset(directory):
    dataset  = load_wiki_dataset(directory,["age","face_location","score"])
    # print "dataset before",len(dataset)
    dataset = dataset.loc[dataset["score"]!=float("-inf")]
    dataset = dataset.reset_index(drop=True)
    print "dataset 1 ",len(dataset)
    dataset = dataset.loc[dataset["score"]!=float("inf")]
    dataset = dataset.reset_index(drop=True)
    print "dataset 2",len(dataset)
    # print dataset
    for i in range(10):
        rand = np.random.randint(0,len(dataset))
        img_file = dataset["file_name"][rand][0]
        image = cv2.imread(os.path.join(directory,img_file))
        print rand

        img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=50)
        rects = []
        for region in regions:
            rects.append(region["rect"])
        bbox = dataset["face_location"][rand]
        draw_rects(image,np.array(rects),np.array(bbox[0]));
        print"score", dataset["score"][rand],type(dataset["score"][rand])
        cv2.imshow(str(dataset["age"][rand])+" years old " + str(dataset["score"][rand]),image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def test_selective_search(img_file):
    for i in range(10):
        image = cv2.imread(img_file)

        img_lbl, regions = selectivesearch.selective_search(image, scale=200, sigma=0.9, min_size=300*(i+1))
        rects = []
        for region in regions:
            rects.append(region["rect"])
        draw_rects(image,rects);


        cv2.imshow("minsize - "+str(50*(i+1))+" total- "+str(len(regions)),image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



