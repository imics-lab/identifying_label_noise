# Function to create model, required for KerasClassifier
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Reshape, Input
from tensorflow.keras.optimizers import RMSprop, SGD

def get_model_1d_cnn(shape_x, shape_y):
    """
    Get a simple CNN for time series estimation.
    :param shape_x:         tuple of ints, Shape of a single image
    :param shape_y:         tuple of ints, Shape of the label
    :return:                Keras model, The compiled model ready for training
    """
    print("buidling 1d cnn X=", shape_x, " y=", shape_y)

    model = Sequential([
        Input(shape=shape_x),
        Reshape((shape_x,1)),
        Conv1D(filters=48, kernel_size=16, activation='relu', padding='same', use_bias=True),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Dropout(0.25),
        Conv1D(filters=48, kernel_size=8, activation='relu', padding='same', use_bias=True),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(shape_y, activation="softmax")
    ])

    model.compile(optimizer='RMSprop', loss='mse', metrics=['accuracy'])

    #model.summary()
    return model
