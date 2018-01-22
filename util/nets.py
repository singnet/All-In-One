
import keras

from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Dense,Flatten,concatenate,Layer
from keras.layers.normalization import  BatchNormalization
from keras.models import Model
from keras import backend as K
from util.preprocess import getAgeGenderDataset
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util.preprocess import load_dataset,getCurrentAgeGenderData,age_gender_dataset_generator


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



def age_loss(y_true,y_pred):
    global LAMDA,SIGMOID
    loss1 = (1-LAMDA) * (1.0/2.0) * K.square(y_pred - y_true);
    loss2 = LAMDA *(1 - K.exp(-(K.square(y_pred - y_true)/(2* SIGMOID))))
    return loss1+loss2
class LambdaUpdateCallBack(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        global LAMDA
        if LAMDA<1:
            LAMDA +=5e-5
        return

    

class AllInOneNeuralNetwork(object):
    def __init__(self,input_shape,learning_rate=1e-4):
        self.input_shape = input_shape
        self.is_built = False
        self.model = self.build()
        self.learning_rate = learning_rate
        self.LOSS_WEIGHTS={
            "age": 0.5,
            "gender" : 0.3,
            "detection": 1,
            "visiblity": 20,
            "pose": 5,
            "landmarks": 100,
            "identity": 0.7,
            "smile": 10
        }
        
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
        age_estimation3 = Dense(1,activation="linear")(age_estimation2)
        age_estimation4 = RoundLayer(name="age_estimation")(age_estimation3)
        # gender probablity

        gender_probablity1 = Dense(1024,activation="relu")(conv6_out_pool_flatten)
        gender_probablity2 = Dense(128,activation="relu")(gender_probablity1)
        gender_probablity3 = Dense(2,activation="softmax",name="gender_probablity")(gender_probablity2)


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
        smile2 = Dense(1,activation="softmax",name="smile")(smile1)

        model = Model(inputs=input_layer,
                        outputs=[detection_probability2,key_point_visibility_2, key_points2,pose2,smile2,
                                gender_probablity3,age_estimation4,face_reco
                                ])
        self.is_built = True;
        return model
    def get_detection_probablity_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[35].output
        model =  Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_key_point_visibility_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[36].output
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_key_points_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[37].output
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_pose_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[38].output
        model =  Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_smile_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[39].output
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_gender_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[40].output
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_age_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[41].output
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = age_loss,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_face_reco_model(self):
        input_layer = self.model.inputs
        output_layer  = self.model.layers[42].output
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss = keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_age_gender_model(self):
        input_layer = self.model.inputs

        # detection_layer = self.model.layers[31].output
        gender_layer  = self.model.layers[40].output
        age_layer  = self.model.layers[41].output
        model = Model(inputs=input_layer,outputs=[age_layer,gender_layer])

        model.compile(loss = [age_loss,keras.losses.categorical_crossentropy],
        loss_weights = [5,1],
        
        optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def get_smile_gender_model(self):
        input_layer = self.model.inputs

        # detection_layer = self.model.layers[31].output
        gender_layer  = self.model.layers[40].output
        smile_layer  = self.model.layers[39].output
        model = Model(inputs=input_layer,outputs=[gender_layer,smile_layer])

        model.compile(loss = [keras.losses.categorical_crossentropy,kers.losses.categorical_crossentropy],
        loss_weights = [1,1],
        
        optimizer=keras.optimizers.Adam(self.learning_rate),metrics=["accuracy"])
        return model
    def train(self):
        # age_gender_dataset = getAgeGenderDataset("/home/mtk/datasets/wiki","/home/mtk/datasets/israel/imdb_crop")
        # X = age_gender_dataset["image"].values
        # age = age_gender_dataset["age"].values
        # gender = age_gender_dataset["gender"].values
        agModel = self.get_age_gender_model()
        agModel.summary()
        # self.model.summary()
        # for i in range(len(self.model.layers)):
        #     print i, self.model.layers[i].name
        # gender = gender.astype(np.uint8)
        # gender = np.eye(2)[gender]
        # age = age.reshape(-1,1)
        
        # X_out = np.zeros((len(X),227,227,3))
        # for i in range(X.shape[0]):
        #     X_out[i] = X[i]
        # # X = X.reshape(-1,227,227,3)`

        # indexes = range(len(X_out))
        # np.random.shuffle(indexes)
        # print indexes
        # X_train  = X_out[indexes[:int(len(indexes)*0.7)]]
        # age_train = age[indexes[:int(len(indexes)*0.7)]]
        # gender_train = gender[indexes[:int(len(indexes)*0.7)]]

        # X_test  = X_out[indexes[int(len(indexes)*0.7):]]
        # age_test = age[indexes[int(len(indexes)*0.7):]]
        # gender_test = gender[indexes[int(len(indexes)*0.7):]]
        # y_train = [age_train,gender_train]
        # y_test = [age_test,gender_test]
        dataset_dir = "/home/mtk/datasets/wiki"
        dataset = load_dataset(dataset_dir,"wiki",["file_name","score","face_location","age","gender"])
        dataset = dataset.loc[dataset["score"]!=float("-inf")]
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.loc[dataset["gender"]!=float("nan")]
        dataset = dataset.reset_index(drop=True)
        print "len of dataset before",len(dataset)
        # dataset = dataset[:15000]
        print "len of dataset after",len(dataset)
        dataset_matrix = dataset.as_matrix()
        
        msk = np.random.rand(len(dataset)) < 0.9
        train_dataset = dataset_matrix[msk]
        test_dataset = dataset_matrix[~msk]
        # training_set = getCurrentAgeGenderData(train_dataset.as_matrix(),"/home/mtk/datasets/wiki")
        print "loading test data"
        testing_set = getCurrentAgeGenderData(test_dataset,dataset_dir)
        print "loaded test data"
        X_test = testing_set["image"].values
        age_test = testing_set["age"].values
        gender_test = testing_set["gender"].values
        
        gender_test = gender_test.astype(np.uint8)
        gender_test = np.eye(2)[gender_test]
        age_test = age_test.reshape(-1,1)
        X_out = np.zeros((len(X_test),227,227,3))
        for i in range(X_test.shape[0]):
            X_out[i] = X_test[i]
        X_test = X_out
        y_test = [age_test,gender_test]
        # agModel.summary()
        agModel.fit_generator(age_gender_dataset_generator(train_dataset,dataset_dir,batch_size=32),epochs = 10,callbacks = [LambdaUpdateCallBack()],steps_per_epoch=1000,validation_data=(X_test,y_test),verbose=True)
        with open("logs.txt","a+") as log_file:
            score = agModel.evaluate(X_test,y_test)
            log_file.write(str(score))
        self.model.save_weights("large_model.h5")
        agModel.save_weights("age_gender_model.h5")
        # self.model.summary()
        