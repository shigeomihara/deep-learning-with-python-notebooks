import keras
from keras import layers

class LSTM:
    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.sequence_length = 120

        print("self.dataFile.num_train_samples=", self.dataFile.num_train_samples)
        print("self.dataFile.raw_data.shape=", self.dataFile.raw_data.shape)

        inputs = keras.Input(shape=(self.sequence_length, self.dataFile.raw_data.shape[-1]))
        lstm = layers.LSTM(16)
        x = lstm(inputs)
        outputs = layers.Dense(1)(x)
        
