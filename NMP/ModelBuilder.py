import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding, TimeDistributed


class ModelBuilder:
    batch_size = None
    seq_length = None
    unique_chars = None

    def __init__(self, bs, sl, uc):
        self.batch_size = bs
        self.seq_length = sl
        self.unique_chars = uc
        self.activation_function = "softmax"


    def build_model(self):
        m = Sequential()

        m.add(
            Embedding(input_dim=self.unique_chars,
                      output_dim=512,
                      batch_input_shape=(self.batch_size, self.seq_length),
                      name="embd_1"
                      )
        )

        m.add(LSTM(256, return_sequences=True, stateful=True, name="lstm_first"))
        m.add(Dropout(0.2, name="drp_1"))

        m.add(LSTM(256, return_sequences=True, stateful=True))
        m.add(Dropout(0.2))

        m.add(LSTM(256, return_sequences=True, stateful=True))
        m.add(Dropout(0.2))

        m.add(TimeDistributed(Dense(self.unique_chars)))
        m.add(Activation(self.activation_function))

        m.load_weights("_Data/Model_Weights/Weights_80.h5", by_name=True)

        return m
