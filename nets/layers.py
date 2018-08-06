from keras.layers import Layer
from keras import backend as K

class RoundLayer(Layer):
    def __init__(self, **kwargs):
        super(RoundLayer, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(RoundLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
