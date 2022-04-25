import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Lambda, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def q_model(input_shape, action_space):
    input_layer = Input(input_shape)
    x = input_layer
    x = Conv2D(64, kernel_size=8, strides=4, activation='relu', input_shape=input_shape,
               padding='same')(x)
    x = Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = Flatten()(x)

    # Hidden Layer he_uniform/he_normal
    x = Dense(512, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dense(256, activation="relu", kernel_initializer='he_uniform')(x)
    #x = Dense(64, activation="relu", kernel_initializer='he_uniform')(x)

    # Output Layer
    output = Dense(len(action_space), activation="linear", kernel_initializer='he_uniform')(x)

    model = Model(inputs=input_layer, outputs=output, name='DQN_CNN')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model

'''class QModel():
    def __init__(self, input_shape, action_space):
        self.model = self.create_model()
        self.input_shape = input_shape
        self.action_space = action_space

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, activation='relu', strides=1, input_shape=(16, 16, 1),
                         padding='same',
                         kernel_initializer='he_normal'))
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same',
                         kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(9, activation='linear', kernel_initializer='he_normal'))
        model.add(Lambda(self.lmd))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model'''
