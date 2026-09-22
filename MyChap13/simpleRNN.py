import keras
from keras import layers

def simpleRNNModel(dataFile):
    print("simpleRNNModel:")
    inputs = keras.Input(shape=(dataFile.sequence_length,
                                dataFile.num_components))
    x = layers.SimpleRNN(16, return_sequences=True)(inputs)
    x = layers.SimpleRNN(16, return_sequences=True)(x)
    x = layers.SimpleRNN(16, return_sequences=False)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "jena_simpleRNN.keras",
            save_best_only=True) ]
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        dataFile.train_dataset,
        epochs=10,
        validation_data=dataFile.val_dataset,
        callbacks=callbacks, )

    model = keras.models.load_model("jena_simpleRNN.keras")

    import numpy as np
    results = model.evaluate(dataFile.test_dataset)
    print(f"  Test RMSE: {np.sqrt(results[0]):.2f}")
    print(f"  Test MAE: {results[1]:.2f}")

    # results = model.evaluate(dataFile.val_dataset)
    # print(f"  Val RMSE: {np.sqrt(results[0]):.2f}")
    # print(f"  Val MAE: {results[1]:.2f}")
    
    # results = model.evaluate(dataFile.train_dataset)
    # print(f"  Train RMSE: {np.sqrt(results[0]):.2f}")
    # print(f"  Train MAE: {results[1]:.2f}")
    
