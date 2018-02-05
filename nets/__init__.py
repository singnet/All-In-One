
import keras

from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Dense,Flatten,concatenate,Layer
from keras.layers.normalization import  BatchNormalization
from keras.models import Model
from keras import backend as K

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util.preprocess import ImdbWikiDatasetPreprocessor
from datetime import datetime
import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from dataset.celeba import CelebAAlignedDataset
from dataset.imdb_wiki import ImdbWikiDataset
import os
from loggers import Log

LAMDA = 0
SIGMOID = 3

class RoundLayer(Layer):
    def __init__(self, **kwargs):
        super(RoundLayer, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def age_margin_mse_loss(y_true,y_pred):
    return K.max(K.square(y_pred -y_true)-2.25,0)

def age_loss(y_true,y_pred):
    
    global LAMDA,SIGMOID
    loss1 = (1-LAMDA) * (1.0/2.0) * K.square(y_pred - y_true);
    loss2 = LAMDA *(1 - K.exp(-(K.square(y_pred - y_true)/(2* SIGMOID))))
    return loss1+loss2
def relative_mse_loss(y_true,y_pred):
    return K.abs(y_true - y_pred)/K.log(y_true)

    # return (1.0/2.0) * K.square(y_pred - y_true)
class LambdaUpdateCallBack(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        global LAMDA
        if LAMDA<1:
            LAMDA +=5e-6
        return
class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.last_loss = 1000000000
        self.last_accuracy = 0
        self.current_model_number = 0;
        self.epoch_number = 0
    # def on_train_begin(self,epoch, logs={}):
    #     return
 
    # def on_train_end(self, logs={}):
    #     return
 
    def on_epoch_begin(self,epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_number+=1
        current_val_loss = logs.get("val_loss")
        current_loss = logs.get("loss")
       

        if (self.last_loss-current_val_loss) > 0.01:
            current_weights_name = "weights"+str(self.current_model_number)+".h5"
            print(" loss improved from "+str(self.last_loss)+" to "+str(current_val_loss)+", Saving model to "+current_weights_name)
            self.model.save_weights("models/"+current_weights_name);
            self.model.save_weights("models/last_weight.h5")
            self.current_model_number+=1
            self.last_loss = current_val_loss
            with open("logs.txt","a+") as logfile:
                logfile.write("________________________________________________________\n")
                logfile.write("EPOCH    =")
                logfile.write(str(epoch)+"\n")
                logfile.write("TRAIN_LOSS =")
                logfile.write(str(current_loss)+"\n")
                logfile.write("VAL_LOSS =")
                logfile.write(str(current_val_loss)+"\n")
                logfile.write("---------------------------------------------------------\n")
                logfile.write("TRAIN_Age_LOSS  =")
                logfile.write(str(logs.get("age_estimation_loss"))+"\n")
                logfile.write("TRAIN_GENDER_LOSS =")
                logfile.write(str(logs.get("gender_probablity_loss"))+"\n")
                logfile.write("---------------------------------------------------------\n")

                logfile.write("TRAIN_Age_ACC  =")
                logfile.write(str(logs.get("age_estimation_acc"))+"\n")
                logfile.write("TRAIN_GENDER_ACC =")
                logfile.write(str(logs.get("gender_probablity_acc"))+"\n")
                logfile.write("---------------------------------------------------------\n")

                logfile.write("VAL_Age_LOSS  =")
                logfile.write(str(logs.get("val_age_estimation_loss"))+"\n")
                logfile.write("VAL_GENDER_LOSS =")
                logfile.write(str(logs.get("val_gender_probablity_loss"))+"\n")
                logfile.write("---------------------------------------------------------\n")

                logfile.write("VAL_Age_ACC  =")
                logfile.write(str(logs.get("val_age_estimation_acc"))+"\n")
                logfile.write("VAL_GENDER_ACC =")
                logfile.write(str(logs.get("val_gender_probablity_acc"))+"\n")

                logfile.write("********************************************************\n")
            with open("epoch_number.json","w+") as json_file:
                data = {"epoch_number":self.epoch_number}
                json.dump(data,json_file,indent=4)


class AllInOneNetwork(object):
    def __init__(self,input_shape,dataset,epochs=10,batch_size=32,learning_rate=1e-4,load_db=False,resume=False,steps_per_epoch=100,large_model_name="large_model",small_model_name="small_model",load_model=None):
        self.input_shape = input_shape
        self.is_built = False
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.LOSS_WEIGHTS={
            "age": 0.5,
            "gender" : 0.3,
            "detection": 1,
            "visiblity": 20,
            "pose": 5,
            "landmarks": 100,
            "identity": 0.7,
            "smile": 10,
            "eye_glasses": 0.5
        }
        self.epochs = epochs
        self.batch_size = batch_size
        self.resume = resume
        self.large_model_name = large_model_name
        self.small_model_name = small_model_name
        self.model = self.build()
        self.save_model_to_json("models/all-in-one-model1.json")
        self.dataset = dataset
        if(load_model!=None and os.path.exists(load_model)):
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading model weights from '"+load_model+"'")
            try:
                self.model.load_weights(load_model)
                Log.DEBUG("Loaded model weights")
            except:
                Log.DEBUG("Unable to load from "+load_model)
            Log.DEBUG_OUT =False
    def save_model_to_json(self,path):
        model_json = self.model.to_json()
        with open(path,"w+") as json_file:
            json_file.write(model_json)
            print "Saved model"
    def build(self):
        input_layer = Input(shape=self.input_shape)
       
        conv1 = Conv2D(96,kernel_size=(11,11),strides=4,activation="relu")(input_layer)
        norm1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(norm1)

        conv2 = Conv2D(256,kernel_size=(5,5),strides=1,padding='same',activation="relu")(pool1)
        norm2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(norm2)
        conv3 = Conv2D(384,kernel_size=(3,3),padding='same')(pool2)

        conv4 = Conv2D(384,kernel_size=(3,3),padding='same')(conv3)
        conv5 = Conv2D(512,kernel_size=(3,3),padding='same')(conv4)


        conv6 = Conv2D(512,kernel_size=(3,3))(conv5)

        conv7 = Conv2D(512,kernel_size=(3,3))(conv6)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
        pool3_flatten = Flatten()(pool3)
        dense1 = Dense(1024,activation="relu")(pool3_flatten)
        dense2 = Dense(512,activation="relu")(dense1)


        face_reco = Dense(10548,activation="softmax",name="face_reco")(dense2)

        # brach from pool1, conv3 and conv5
        pool1_out_conv = Conv2D(256,kernel_size=(4,4),strides=4,activation="relu")(pool1) 
        conv3_out_conv = Conv2D(256,kernel_size=(2,2),strides=2,activation="relu")(conv3)
        conv5_out_pool = MaxPooling2D(pool_size=(2, 2))(conv5)
        # print "pool1_out_conv",pool1_out_conv.shape
        # print "conv3_out_conv",conv3_out_conv.shape
        # print "conv5_out_pool",conv5_out_pool.shape
        # subject independant layers
        merge_1 = concatenate([pool1_out_conv,conv3_out_conv,conv5_out_pool])
        merge_1_conv = Conv2D(256,kernel_size=(1,1),strides=(1, 1),activation="relu")(merge_1)
        merge_1_conv_flatten = Flatten()(merge_1_conv)
        merge_1_dense = Dense(2048,activation="relu")(merge_1_conv_flatten)

        #subject dependant layers
        conv6_out_pool = MaxPooling2D(pool_size=(2, 2))(conv6)
        conv6_out_pool_flatten = Flatten()(conv6_out_pool)
        # age estimation layers
        age_estimation1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        age_estimation2 = Dense(128,activation="relu")(age_estimation1)
        age_estimation3 = Dense(1,activation="linear",name="age_estimation")(age_estimation2)
        # age_estimation4 = RoundLayer(name="age_estimation")(age_estimation3)
        # gender probablity

        gender_probablity1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        gender_probablity2 = Dense(128,activation="relu")(gender_probablity1)
        gender_probablity3 = Dense(2,activation="softmax",name="gender_probablity")(gender_probablity2)

       

        # Young
        young_1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        young_2 = Dense(128,activation="relu")(young_1)
        young_3 = Dense(2,activation="softmax",name="is_young")(young_2)

        #
        

        # face detection    
        detection_probability1 = Dense(512,activation="relu")(merge_1_dense)
        detection_probability2 = Dense(2,activation="softmax",name="detection_probablity")(detection_probability1)

        # key points(21) visibility probablity

        key_point_visibility_1 = Dense(512,activation="relu")(merge_1_dense)
        key_point_visibility_2 = Dense(21,activation="linear",name="kpoints_visibility")(key_point_visibility_1)

        # key points(21) location point(x,y) visibility probablity
        key_points1 = Dense(512,activation="relu")(merge_1_dense)
        key_points2 = Dense(42,activation="linear",name="key_points")(key_points1)

        # Pose value of the face(roll,pitch,yaw)
        pose1 = Dense(512,activation="relu")(merge_1_dense)
        pose2 = Dense(3,activation="linear",name="pose")(pose1)

        # probablity face being smile face
        smile1 = Dense(512,activation="relu")(merge_1_dense)
        smile2 = Dense(2,activation="softmax",name="smile")(smile1)

        # probablity face being smile face
        eye_glasses1 = Dense(512,activation="relu")(merge_1_dense)
        eye_glasses2 = Dense(2,activation="softmax",name="eye_glasses")(eye_glasses1)
        
        # probablity face being smile face
        mouse_slightly_open1  = Dense(512,activation="relu")(merge_1_dense)
        mouse_slightly_open2 = Dense(2,activation="softmax",name="mouse_slightly_open")(mouse_slightly_open1)
        

        model = Model(inputs=input_layer,
                        outputs=[detection_probability2,key_point_visibility_2, key_points2,pose2,smile2,
                                gender_probablity3,age_estimation3,face_reco,young_3,eye_glasses2,
                                mouse_slightly_open2
                                ])
        
        self.is_built = True;
        return model
    def get_layer(self,name):
        for layer in self.model.layers:
            if layer.name == name:
                return layer
        raise Exception("Layer with name "+name + " does not exist")
    """Get model which have output layer given by labels.
    Parameters
    ----------
    labels : list
        list of output labels. Elements of the list should be one or more 
        of ["detection_probablity","kpoints_visibility","key_points","pose","smile",
            "gender_probablity","age_estimation","face_reco","is_young","eye_glasses",
            "mouse_slightly_open"]

    """

    def get_model_with_labels(self,labels):
        all_lists = ["detection_probablity","kpoints_visibility","key_points","pose","smile",
            "gender_probablity","age_estimation","face_reco","is_young","eye_glasses",
            "mouse_slightly_open"]
        assert type(labels) == list, " argment should be list type"
        assert not(labels is None or len(labels)==0), "Labels should not be empty"
        assert set(labels).issubset(all_lists), str(labels)+" contains lists which are not in "+ str(all_lists)
        
        input_layer = self.model.inputs
        output_layers = []
        for label in labels:
            output_layer  = self.get_layer(label)
            output_layers.append(output_layer.output)
        model =  Model(inputs=input_layer,outputs=output_layers)
        return model
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
        with open("log.txt","a+") as logfile:
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
        ageGenderModel.compile(loss = [relative_mse_loss, keras.losses.categorical_crossentropy],loss_weights=[0.01,1],optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        ageGenderModel.summary()

        X_test = self.dataset.test_dataset_images
        age_test = self.dataset.test_dataset["Age"].as_matrix()
        gender_test = self.dataset.test_dataset["Gender"].as_matrix().astype(np.uint8)
        gender_test = np.eye(2)[gender_test]
        y_test = [age_test,gender_test]
        if self.resume:
            checkPoint = self.resume_model()
            callbacks = [checkPoint]
        else:
            callbacks = [CustomModelCheckPoint()]
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
    def train(self):
        if type(self.dataset) == CelebAAlignedDataset:
            self.train_celebA()
        elif type(self.dataset) == ImdbWikiDataset:
            self.train_imdb_wiki()
        