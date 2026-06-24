import keras
from keras import layers
import pickle
import matplotlib.pyplot as plt

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
        self.history = self.model.fit(
            self.dataFile.train_dataset,
            epochs=10,
            validation_data=self.dataFile.val_dataset,
            callbacks=self.callbacks,
        )

    def saveHistory(self):
        f = open("trainingHistory.dat", 'wb')
        pickle.dump(self.history.history, f)
        f.close()

    def plotSavedHistory(self):
        f = open("trainingHistory.dat", 'rb')
        history = pickle.load(f)
        f.close()

        loss = history["mae"]
        val_loss = history["val_mae"]
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, "r--", label="Training MAE")
        plt.plot(epochs, val_loss, "b", label="Validation MAE")
        plt.title("Training and validation MAE")
        plt.legend()
        plt.show()

        
    
