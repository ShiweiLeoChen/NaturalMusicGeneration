from NMP import charIndex_json, data_directory, model_weights_directory
from NMP import BATCH_SIZE, SEQ_LENGTH
from NMP.BatchReader import BatchReader
from NMP.ModelBuilder import ModelBuilder
from tqdm import tqdm
import os
import json
import numpy as np


class ModelTrainer:
    data = None
    epochs = None

    def __init__(self, d, e):
        self.data = d
        self.epochs = e

    def train_model(self):
        char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(self.data))))}

        with open(os.path.join(data_directory, charIndex_json), mode="w") as f:
            json.dump(char_to_index, f)

        unique_chars = len(char_to_index)

        m = ModelBuilder(BATCH_SIZE, SEQ_LENGTH, unique_chars)
        model = m.build_model()
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        all_characters = np.asarray([char_to_index[c] for c in self.data], dtype=np.int32)

        b = BatchReader(all_characters, unique_chars)

        for epoch in tqdm(range(self.epochs)):
            for i, (x, y) in enumerate(b.read_batches()):
                model.train_on_batch(x, y)
            if (epoch + 1) % 10 == 0:
                if not os.path.exists(model_weights_directory):
                    os.makedirs(model_weights_directory)
                model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch + 1)))
                print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch + 1, epoch + 1))
