import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Lambda, Conv2D

class QModel():
    def __init__(self, input_shape, action_space):
        self.model = self.create_model()
        self.input_shape = input_shape
        self.action_space = action_space

    def create_model(self):
        model = Sequential()

        return model
