from NMP import data_directory, data_file
from NMP.ModelTrainer import ModelTrainer
import os

epoch = 90

file = open(os.path.join(data_directory, data_file), mode='r')
data = file.read()
file.close()

m = ModelTrainer(data, epoch)

if __name__ == "__main__":
    m.train_model()
