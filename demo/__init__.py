from keras.models import model_from_json,Model
import os
from scipy.io import loadmat
import cv2
import numpy as np
import pandas as pd


def get_layer(model,name):
    for layer in model.layers:
        if layer.name == name:
            return layer
    raise Exception("Layer with name "+name + " does not exist")
def load_model(model_json_path,model_h5_path,layer_names):
    with open(model_json_path,"r") as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(model_h5_path)
        layer_output = []
        for lname in layer_names:
            layer_output+= [get_layer(model,lname).output]

        output = Model(inputs=model.inputs,output=layer_output)
        return output
def selective_search_demo():
    dataset_dir = "/home/mtk/datasets/DataSet"
    b_box = loadmat(os.path.join(dataset_dir,"BoundingBox.mat"))
    bboxesT = b_box["bboxesT"]
    bboxesTr = b_box["bboxesTr"].astype(np.uint8)
    while True:
        index = np.random.randint(0,len(bboxesTr))
        image = cv2.imread(os.path.join(dataset_dir, "TrainImages",str(index+1)+".jpg"))
        bbox = bboxesTr[index]
        image = cv2.rectangle(image,(bbox[0],bbox[0]+ bbox[2]),(bbox[1],bbox[1]+bbox[3]),(255,255,0))
        # print
        cv2.imshow("Image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print b_box
