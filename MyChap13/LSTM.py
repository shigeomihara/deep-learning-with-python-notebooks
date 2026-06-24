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
        
        self.model = keras.Model(inputs, outputs)

        self.callbacks = [
            keras.callbacks.ModelCheckpoint("jena_lstm.keras", save_best_only=True)
        ]
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def fit(self):
        history = self.model.fit(
            self.dataFile.train_dataset,
            epochs=10,
            validation_data=self.dataFile.val_dataset,
            callbacks=self.callbacks,
        )
