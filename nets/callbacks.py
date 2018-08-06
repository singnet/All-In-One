import keras
import json

from nets.loss_functions import LAMDA

class LambdaUpdateCallBack(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        global LAMDA
        if LAMDA<1:
            LAMDA +=5e-5
        return
class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.last_loss = 1000000000
        self.last_accuracy = 0
        self.current_model_number = 0;
        self.epoch_number = 0


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
            with open("logs/logs.txt","a+") as logfile:
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
