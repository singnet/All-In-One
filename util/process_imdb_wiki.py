from scipy.io import loadmat
import os
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import selectivesearch


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    name = meta[db][0, 0]["name"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    face_location = meta[db][0,0]["face_location"][0]
    data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
            "second_score": second_face_score,"name":name,"face_location":face_location}
    dataset = pd.DataFrame(data)
    return dataset


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1
def load_dataset(dataset_dir,dataset_type,keys):
    if dataset_type=="wiki":
        return load_wiki_dataset(dataset_dir,keys)
    elif dataset_type == "imdb":
        return load_imdb_dataset(dataset_dir,keys)
def load_imdb_dataset(directory,keys):
    dataset = get_meta(os.path.join(directory,"imdb.mat"),"imdb")
    output = pd.DataFrame()
    for key in keys:
        output[key] = dataset[key]
    if not "file_name" in keys:     
        output["file_name"] = dataset["file_name"]
    return output
def load_wiki_dataset(directory,keys):
    # keys one or more of [file_name,gender,age,score,second_score,name]
    # file name will be included by default
    dataset = get_meta(os.path.join(directory,"wiki.mat"),"wiki")
    output = pd.DataFrame()
    for key in keys:
        output[key] = dataset[key]
    if not "file_name" in keys:     
        output["file_name"] = dataset["file_name"]
    return output