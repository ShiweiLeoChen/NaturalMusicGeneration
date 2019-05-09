from NMP import BATCH_SIZE, SEQ_LENGTH
import numpy as np


class BatchReader:
    all_chars = None
    unique_chars = None

    def __init__(self, ac, uc):
        self.all_chars = ac
        self.unique_chars = uc

    def read_batches(self):
        length = self.all_chars.shape[0]
        batch_chars = int(length / BATCH_SIZE)

        for start in range(0, batch_chars - SEQ_LENGTH, 64):
            X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
            Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, self.unique_chars))
            for batch_index in range(0, 16):
                for i in range(0, 64):
                    X[batch_index, i] = self.all_chars[batch_index * batch_chars + start + i]
                    Y[batch_index, i, self.all_chars[batch_index * batch_chars + start + i + 1]] = 1
            
            yield X, Y
