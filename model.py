from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense


def build_model(input_shape, n_output):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape))
    model.add(Dense(n_output, activation="softmax"))

    return model
