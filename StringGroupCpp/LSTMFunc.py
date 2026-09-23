import keras
from keras import layers

def LSTMModel(dataFile):
    print("LSTMModel:")
    inputs = keras.Input(shape=(dataFile.sequence_length,
                                dataFile.num_components))
    x = layers.LSTM(16, return_sequences=True)(inputs)
    x = layers.LSTM(16, return_sequences=True)(x)
    x = layers.LSTM(16)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "jena_lstm.keras",
            save_best_only=True) ]
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        dataFile.train_dataset,
        epochs=100,
        validation_data=dataFile.val_dataset,
        callbacks=callbacks, )

    model = keras.models.load_model("jena_lstm.keras")

    import numpy as np
    results = model.evaluate(dataFile.test_dataset)
    print(f"  Test RMSE: {np.sqrt(results[0]):.4f}")
    print(f"  Test MAE: {results[1]:.4f}")

    import matplotlib.pyplot as plt
    
    loss = history.history["mae"]
    val_loss = history.history["val_mae"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "r--", label="Training MAE")
    plt.plot(epochs, val_loss, "b", label="Validation MAE")
    plt.title("Training and validation MAE")
    plt.legend()
    plt.show()
    
    # results = model.evaluate(dataFile.val_dataset)
    # print(f"  Val RMSE: {np.sqrt(results[0]):.2f}")
    # print(f"  Val MAE: {results[1]:.2f}")
    
    # results = model.evaluate(dataFile.train_dataset)
    # print(f"  Train RMSE: {np.sqrt(results[0]):.2f}")
    # print(f"  Train MAE: {results[1]:.2f}")
    
