
import keras

from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Dense,Flatten,concatenate,Layer
from keras.layers.normalization import  BatchNormalization
from keras.models import Model
from keras import backend as K

from sklearn.model_selection import train_test_split
import json
from nets.layers import RoundLayer

class AllInOneModel(object):
    def __init__(self,input_shape):
        self.is_built = False
        self.input_shape = input_shape
        self.model = self.build()
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
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(512,activation="relu")(dropout1)
        dropout2 = Dropout(0.2)(dense2)

        face_reco = Dense(10548,activation="softmax",name="face_reco")(dropout2)
        # branch from pool1, conv3 and conv5
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
        merge_1_dropout = Dropout(0.2)(merge_1_dense)

        #subject dependant layers
        conv6_out_pool = MaxPooling2D(pool_size=(2, 2))(conv6)
        conv6_out_pool_flatten = Flatten()(conv6_out_pool)
        # age estimation layers
        age_estimation1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        age_drop1 = Dropout(0.2)(age_estimation1)
        age_estimation2 = Dense(128,activation="relu")(age_drop1)
        age_drop2  = Dropout(0.2)(age_estimation2)
        age_estimation3 = Dense(1,activation="linear")(age_drop2)
        age_estimation4 = RoundLayer(name="age_estimation")(age_estimation3)
        # gender probablity

        gender_probablity1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        gender_drop1 = Dropout(0.2)(gender_probablity1)
        gender_probablity2 = Dense(128,activation="relu")(gender_drop1)
        gender_drop2 = Dropout(0.2)(gender_probablity2)
        gender_probablity3 = Dense(2,activation="softmax",name="gender_probablity")(gender_drop2)

        # Young
        young_1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        young_drop1 = Dropout(0.2)(young_1)
        young_2 = Dense(128,activation="relu")(young_drop1)
        young_drop2 = Dropout(0.2)(young_2)
        young_3 = Dense(2,activation="softmax",name="is_young")(young_drop2)
        #

        # face detection
        detection_probability1 = Dense(512,activation="relu")(merge_1_dropout)
        detection_probability_drop =  Dropout(0.2)(detection_probability1)
        detection_probability2 = Dense(2,activation="softmax",name="detection_probablity")(detection_probability_drop)

        # key points(21) visibility probablity

        key_point_visibility_1 = Dense(512,activation="relu")(merge_1_dropout)
        key_point_visibility_drop = Dropout(0.2)(key_point_visibility_1)
        key_point_visibility_2 = Dense(21,activation="linear",name="key_points_visibility")(key_point_visibility_drop)

        # key points(21) location point(x,y) visibility probablity
        key_points1 = Dense(512,activation="relu")(merge_1_dropout)
        key_points_drop = Dropout(0.2)(key_points1)
        key_points2 = Dense(42,activation="linear",name="key_points")(key_points_drop)

        # Pose value of the face(roll,pitch,yaw)
        pose1 = Dense(512,activation="relu")(merge_1_dropout)
        pose_drop = Dropout(0.2)(pose1)
        pose2 = Dense(3,activation="linear",name="pose")(pose_drop)

        # probablity face being smile face
        smile1 = Dense(512,activation="relu")(merge_1_dropout)
        smile_drop = Dropout(0.2)(smile1)
        smile2 = Dense(2,activation="softmax",name="smile")(smile_drop)

        # probablity face being smile face
        eye_glasses1 = Dense(512,activation="relu")(merge_1_dropout)
        eye_glasses_drop = Dropout(0.2)(eye_glasses1)
        eye_glasses2 = Dense(2,activation="softmax",name="eye_glasses")(eye_glasses_drop)

        # probablity face being smile face
        mouse_slightly_open1  = Dense(512,activation="relu")(merge_1_dropout)
        mouse_slightly_open_drop = Dropout(0.2)(mouse_slightly_open1)
        mouse_slightly_open2 = Dense(2,activation="softmax",name="mouse_slightly_open")(mouse_slightly_open_drop)

        model = Model(inputs=input_layer,
                        outputs=[detection_probability2,key_point_visibility_2, key_points2,pose2,smile2,
                                gender_probablity3,age_estimation4,face_reco,young_3,eye_glasses2,
                                mouse_slightly_open2
                                ])

        self.is_built = True;
        return model

    def save_model_to_json(self,path):
        model_json = self.model.to_json()
        with open(path,"w+") as json_file:
            json_file.write(model_json)
            print ("Saved model")

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
