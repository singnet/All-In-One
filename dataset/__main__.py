from dataset.aflw import AflwDataset
import cv2
import os
import dlib


def main():
    aflwDataset= AflwDataset("",None)
    dataset_dir = "/home/samuel/dataset/aflw/aflw/data/flickr/0/"
    for img_file in os.listdir(dataset_dir):
        img = cv2.imread(os.path.join(dataset_dir,img_file))
        aflwDataset.selective_search(img)


if __name__ == "__main__":
    main()
