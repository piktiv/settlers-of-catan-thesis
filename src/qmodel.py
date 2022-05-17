import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Lambda, Conv2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def lmd(input):
    v = input[:, 0]  # shape 32
    v = tf.reshape(v, [-1, 1])
    a = input[:, 1:]  # shape 32,8
    a_mean = tf.math.reduce_mean(a, axis=1)
    a_mean = tf.reshape(a_mean, [-1, 1])
    Q = v + (a - a_mean)
    return Q

def q_model(input_shape, action_space):
    '''model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=5, activation='relu', strides=1, input_shape=(23, 21, 1),
                     padding='same',
                     kernel_initializer='he_normal'))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same',
                     kernel_initializer='he_normal'))
    model.add(Flatten())
    model.add(Dense(512, activation="relu", kernel_initializer='he_normal'))
    model.add(Dense(256, activation="relu", kernel_initializer='he_normal'))
    model.add(Dense(len(action_space), activation='linear', kernel_initializer='he_normal'))
    #model.add(Lambda(lmd))
    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])
    model.summary()

    return model'''



    input_layer = Input(input_shape)
    x = input_layer
    x = Conv2D(64, kernel_size=8, strides=4, activation='relu', input_shape=input_shape,
               padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = Flatten()(x)

    # Hidden Layer he_uniform/he_normal
    x = Dense(512, activation="relu", kernel_initializer='he_normal')(x)
    # Hidden layer mean(input, output) = 389 nodes
    #x = Dense(389, activation="relu", kernel_initializer='he_normal')(x)
    x = Dense(256, activation="relu", kernel_initializer='he_normal')(x)
    #x = BatchNormalization()(x)  # Before linear expression
    #x = Dense(64, activation="relu", kernel_initializer='he_uniform')(x)

    # Output Layer
    output = Dense(len(action_space), activation="linear", kernel_initializer='he_normal')(x)

    model = Model(inputs=input_layer, outputs=output, name='DQN_CNN')

    # Try lowering learning_rate 5e-5
    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model

