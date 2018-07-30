from keras import backend as K

LAMDA = 0
SIGMOID = 3
def age_margin_mse_loss(y_true,y_pred):
    return K.max(K.square(y_pred -y_true)-2.25,0)

def age_loss(y_true,y_pred):

    global LAMDA,SIGMOID
    loss1 = (1-LAMDA) * (1.0/2.0) * K.square(y_pred - y_true);
    loss2 = LAMDA *(1 - K.exp(-(K.square(y_pred - y_true)/(2* SIGMOID))))
    return loss1+loss2
def relative_mse_loss(y_true,y_pred):
    return K.square(y_true - y_pred)/K.sqrt(y_true)
