
import keras

from keras import backend as K

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from keras.backend.tensorflow_backend import set_session
from dataset.celeba import CelebAAlignedDataset
from dataset.imdb_wiki import ImdbWikiDataset
from dataset.aflw import AflwDataset
import os
from loggers import Log
from nets.model import AllInOneModel
from   util import DatasetType
from nets.callbacks import LambdaUpdateCallBack,CustomModelCheckPoint
from nets.loss_functions import age_loss


class AllInOneNetwork(object):
    def __init__(self,config):
        self.config = config
        self.model = AllInOneModel(self.config.image_shape)
        if(config.model_weight!=None and os.path.exists(config.model_weight)):
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading model weights from '"+config.model_weight+"'")
            try:
                self.model.model.load_weights(config.model_weight)
                Log.DEBUG("Loaded model weights")
            except:
                Log.DEBUG("Unable to load model weight from "+config.model_weight)
            Log.DEBUG_OUT =False

    def resume_model(self):
        customCheckPoint = CustomModelCheckPoint()
        REMAINING_EPOCHS = self.epochs
        if os.path.exists("epoch_number.json"):
            with open("epoch_number.json","r") as json_file:
                try:
                    data = json.load(json_file)
                    customCheckPoint.epoch_number = data["epoch_number"]
                    REMAINING_EPOCHS -= customCheckPoint.epoch_number
                except:
                    print ("unable to read epoch number from file. resuming epoch from 0.")

            print("resuming from previous epoch number:")
            print("previous epoch number",customCheckPoint.epoch_number)
            print("remaining epochs",REMAINING_EPOCHS)
        if REMAINING_EPOCHS < 0:
            REMAINING_EPOCHS =1
        with open("logs.txt","a+") as logfile:
            str_date = datetime.now().strftime("%d, %b, %Y %H:%M:%S")
            logfile.write("Starting to train model\n")
            logfile.write("Dataset :"+self.preprocessor.dataset_type+"\n")
            logfile.write(str_date+"\n")
        return customCheckPoint

    def train_imdb_wiki(self):
        if not self.dataset.dataset_loaded:
            self.dataset.load_dataset()
        assert self.dataset.dataset_loaded ==True, "Dataset is not loaded"
        ageGenderModel = self.get_model_with_labels(["age_estimation","gender_probablity"])
        if self.freeze:
            for i in range(len(ageGenderModel.layers)):
                if ageGenderModel.layers[i].name in ["age_estimation","gender_probablity","dense_4","dense_5","dense_6","dense_7"] :
                    ageGenderModel.layers[i].trainable = True
                    print ageGenderModel.layers[i].name, "trainable == True"
                else:
                    print ageGenderModel.layers[i], "trainable  == False"
                    ageGenderModel.layers[i].trainable = False
        ageGenderModel.compile(loss = [age_loss, keras.losses.categorical_crossentropy],loss_weights=[self.LOSS_WEIGHTS["age"],self.LOSS_WEIGHTS["gender"]],optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        ageGenderModel.summary()

        X_test = self.dataset.test_dataset_images
        age_test = self.dataset.test_dataset["Age"].as_matrix()
        gender_test = self.dataset.test_dataset["Gender"].as_matrix().astype(np.uint8)
        gender_test = np.eye(2)[gender_test]
        y_test = [age_test,gender_test]
        if self.resume:
            checkPoint = self.resume_model()
            callbacks = [checkPoint,LambdaUpdateCallBack()]
        else:
            callbacks = [CustomModelCheckPoint(),LambdaUpdateCallBack()]

        ageGenderModel.fit_generator(self.dataset.generator(batch_size=self.batch_size),epochs = self.epochs,callbacks = callbacks,steps_per_epoch=self.steps_per_epoch,validation_data=(X_test,y_test),verbose=True)
        with open("logs/logs.txt","a+") as log_file:
            score = ageGenderModel.evaluate(X_test,y_test)
            log_file.write(str(score))
        self.model.save_weights("models/"+self.large_model_name+".h5")
        ageGenderModel.save_weights("models/"+self.small_model_name+".h5")

    def train_celebA(self):
        if not self.dataset.dataset_loaded:
            self.dataset.load_dataset()
        assert self.dataset.dataset_loaded ==True, "Dataset is not loaded"
        smileModel = self.get_model_with_labels(["smile"])
        if self.freeze:
            for i in range(len(smileModel.layers)):
                if smileModel.layers[i].name not in ["smile" ,"dense_14"]:
                    smileModel.layers[i].trainable=False
                else:
                    smileModel.layers[i].trainable = True
        smileModel.compile(loss = keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        smileModel.summary()

        X_test = self.dataset.test_dataset_images
        smiling = self.dataset.test_dataset["Smiling"].as_matrix().astype(np.uint8)
        y_test = np.eye(2)[smiling]
        if self.resume:
            checkPoint = self.resume_model()
            callbacks = [checkPoint]
        else:
            callbacks = [CustomModelCheckPoint()]

        smileModel.fit_generator(self.dataset.generator(batch_size=self.batch_size),epochs = self.epochs,callbacks = callbacks,steps_per_epoch=self.steps_per_epoch,validation_data=(X_test,y_test),verbose=True)
        with open("logs/logs.txt","a+") as log_file:
            score = smileModel.evaluate(X_test,y_test)
            log_file.write(str(score))
        self.model.save_weights("models/"+self.large_model_name+".h5")
        smileModel.save_weights("models/"+self.small_model_name+".h5")

    def save_model(self,model,score):
        self.model.model.save_weights("models/"+self.config.large_model_name+".h5")
        model.save_weights("models/"+self.config.small_model_name+".h5")
        print("Score:",score)
        with open("logs/logs.txt", "a+") as log_file:
            log_file.write("----------------------------------------\n")
            log_file.write("large model _name:"+(self.config.large_model_name)+"\n")
            log_file.write("small model _name:"+(self.config.small_model_name)+"\n")
            log_file.write("Score: "+str(score))
            log_file.write("________________________________________\n")

    def train_age_network(self):
        age_model = self.model.get_model_with_labels(["age_estimation"])
        dataset = self.getDatasetFromString(self.config)
        if not dataset.dataset_loaded:
            dataset.load_dataset()
        X_test = dataset.test_dataset_images
        X_test = X_test.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
        age_test = dataset.test_dataset["Age"].as_matrix()
        age_model.compile(loss = age_loss,optimizer=keras.optimizers.Adam(self.config.getLearningRate()),metrics=["accuracy"])
        callbacks = [LambdaUpdateCallBack()]
        age_model.summary()
        age_model.fit_generator(dataset.age_data_genenerator(self.config.batch_size),
                epochs = self.config.epochs,
                steps_per_epoch = self.config.steps_per_epoch,
                validation_data = [X_test,age_test],
                callbacks = callbacks
        )
        score = age_model.evaluate(X_test,age_test)
        self.save_model(age_model,score)


    def train_gender_network(self):
        gender_model = self.model.get_model_with_labels(["gender_probablity"])
        dataset = self.getDatasetFromString(self.config)
        if not dataset.dataset_loaded:
            dataset.load_dataset()
        X_test = dataset.test_dataset_images
        X_test = X_test.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
        gender_test = dataset.test_dataset["Gender"].as_matrix().astype(np.uint8)
        gender_test = np.eye(2)[gender_test]
        gender_model.compile(loss = keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(self.config.getLearningRate()),metrics=["accuracy"])
        callbacks = None
        gender_model.summary()
        gender_model.fit_generator(dataset.gender_data_genenerator(self.config.batch_size),
                epochs = self.config.epochs,
                steps_per_epoch = self.config.steps_per_epoch,
                validation_data = [X_test,gender_test],
                callbacks = callbacks
        )
        score = gender_model.evaluate(X_test,gender_test)
        self.save_model(gender_model,score)

    def train_smile_network(self):
        smile_model = self.model.get_model_with_labels(["smile"])
        dataset = self.getDatasetFromString(self.config)
        if not dataset.dataset_loaded:
            dataset.load_dataset()
        X_test = dataset.test_dataset_images
        X_test = X_test.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
        smile_test = dataset.test_dataset["Smiling"].as_matrix().astype(np.uint8)
        smile_test  = np.eye(2)[smile_test]
        smile_model.compile(loss = keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(self.config.getLearningRate()),metrics=["accuracy"])
        callbacks = None
        smile_model.summary()
        smile_model.fit_generator(dataset.smile_data_generator(self.config.batch_size),
                epochs = self.config.epochs,
                steps_per_epoch = self.config.steps_per_epoch,
                validation_data = [X_test,smile_test],
                callbacks = callbacks
        )
        score = smile_model.evaluate(X_test,smile_test)
        self.save_model(smile_model,score)

    def getDatasetFromString(self, config):
        if self.config.dataset.lower() == "celeba":
            return CelebAAlignedDataset(self.config)
        elif self.config.dataset.lower() == "wiki" or self.config.dataset.lower()=="imdb":
            return ImdbWikiDataset(self.config)
        elif self.config.dataset.lower() == "aflw":
            return AflwDataset(self.config)
        else:
            raise NotImplementedError("Not implemented for "+str(config.dataset))

    def train_face_detection_network(self):
        face_detection_model = self.model.get_model_with_labels(["detection_probablity"])
        dataset = self.getDatasetFromString(self.config)
        if not dataset.dataset_loaded:
            dataset.load_dataset()
        X_test = dataset.test_dataset_images
        X_test = X_test.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
        detection_test = dataset.test_detection
        detection_test = np.eye(2)[detection_test]
        face_detection_model.compile(loss = keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(self.config.getLearningRate()),metrics=["accuracy"])
        callbacks = None
        face_detection_model.summary()
        face_detection_model.fit_generator(dataset.detection_data_genenerator(self.config.batch_size),
                epochs = self.config.epochs,
                steps_per_epoch = self.config.steps_per_epoch,
                validation_data = [X_test,detection_test],
                callbacks = callbacks
        )
        score = face_detection_model.evaluate(X_test,detection_test)
        self.save_model(face_detection_model,score)
    def train_key_points_localization_network(self):
        pass
    def train_key_points_visiblity_network(self):
        pass
    def train_pose_network(self):
        pass
    def train_face_recognition_network(self):
        pass
    def train(self):
        label = self.config.label.lower()
        if label == "age":
            self.train_age_network()
        elif label == "gender":
            self.train_gender_network()
        elif label == "smile":
            self.train_smile_network()
        elif label == "detection":
            self.train_face_detection_network()
        elif label == "key_points":
            self.train_key_points_localization_network()
        elif label == "key_points_visiblity":
            self.train_key_points_visiblity_network()
        elif label == "identification":
            self.train_face_recognition_network()
        elif label == "pose":
            self.train_pose_network()
        else:
            raise NotImplemented("Traing method not implemented for "+str(label))
