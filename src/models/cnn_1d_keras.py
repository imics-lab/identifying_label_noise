# Function to create model, required for KerasClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import SGD

def get_model_1d_cnn(shape_x, shape_y):
    """
    Get a simple CNN for image estimation.
    :param shape_x:         tuple of ints, Shape of a single image
    :param shape_y:         tuple of ints, Shape of the label
    :return:                Keras model, The compiled model ready for training
    """
    print("buidling 1d cnn")

    model = Sequential()
    model.add(Conv1D(filters=48, kernel_size=16, input_shape=shape_x, activation='relu', padding='same', use_bias=True))
    model.add(MaxPooling2D(pool_size=(4)))
    model.add(Conv1D(filters=48, kernel_size=8, activation='relu', padding='same', use_bias=True))
    model.add(MaxPooling2D(pool_size=(4)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(shape_y, activation="sigmoid"))

    model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['accuracy'])
    return model
