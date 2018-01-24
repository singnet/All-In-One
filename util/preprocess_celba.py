import numpy as np
import pandas as pd
import os
import cv2


def  load_dataset_annotation(anno_file_path,bounding_box_path,keys=["Male","Smiling"]):
    header_cols = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    df = pd.read_csv(anno_file_path, sep=r"\s*", header=1, names=header_cols)
    output = df[keys]
    if keys.count("Male")>0:
        output["Male"] = 1/2.0 * output["Male"] + 1/2.0
    if keys.count("Smiling"):
        output["Smiling"] = 1/2.0 * output["Smiling"] + 1/2.0
    return output
def load_images(datatset_dir,dataset_annotation):
    output = pd.DataFrame("images")
    # for i in range(len(dataset_annotation)):
    #     output.loc[-1] = [face_image, matrix[index][3], matrix[index][4]]  # adding a row
    #     output.index = output.index + 1  # shifting index
    #     output = output.sort_index()
def load_dataset(dataset_dir,anno_file_path,keys):
    pass

