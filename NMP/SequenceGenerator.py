from NMP import data_directory, model_weights_directory, charIndex_json
from NMP.ModelMaker import Maker
import numpy as np
import os
import json


class Generator:
    epoch_num = 0
    initial_index = 0
    seq_length = 0

    def __init__(self, en, ii, sl):
        self.epoch_num = en
        self.initial_index = ii
        self.seq_length = sl

    def generate_sequence(self):
        with open(os.path.join(data_directory, charIndex_json)) as f:
            char_to_index = json.load(f)
        index_to_char = {i: ch for ch, i in char_to_index.items()}
        unique_chars = len(index_to_char)

        m = Maker(unique_chars)
        model = m.make_model()
        model.load_weights(model_weights_directory + "Weights_{}.h5".format(self.epoch_num))

        sequence_index = [self.initial_index]

        for _ in range(self.seq_length):
            batch = np.zeros((1, 1))
            batch[0, 0] = sequence_index[-1]

            predicted_probs = model.predict_on_batch(batch).ravel()
            sample = np.random.choice(range(unique_chars), size=1, p=predicted_probs)

            sequence_index.append(sample[0])

        sequence = ''.join(index_to_char[c] for c in sequence_index)
        cnt = 0
        for i in sequence:
            cnt += 1
            if i == "\n":
                break
        sequence_temp = sequence[cnt:]

        cnt = 0
        for i in sequence_temp:
            cnt += 1
            if i == "\n" and sequence_temp[cnt] == "\n":
                break
        result = sequence_temp[:cnt]

        return result
