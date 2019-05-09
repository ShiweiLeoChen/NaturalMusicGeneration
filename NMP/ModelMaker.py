import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding


class Maker:
    unique_chars = ''
    activation_function = "softmax"

    def __init__(self, uc):
        self.unique_chars = uc

    def make_model(self):
        m = Sequential()

        m.add(Embedding(input_dim=self.unique_chars, output_dim=512, batch_input_shape=(1, 1)))

        m.add(LSTM(256, return_sequences=True, stateful=True))
        m.add(Dropout(0.2))

        m.add(LSTM(256, return_sequences=True, stateful=True))
        m.add(Dropout(0.2))

        m.add(LSTM(256, stateful=True))
        m.add(Dropout(0.2))

        m.add((Dense(self.unique_chars)))
        m.add(Activation(self.activation_function))

        return m
