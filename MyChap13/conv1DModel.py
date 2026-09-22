import keras
from keras import layers

def conv1DModel(dataFile):
    print("conv1DModel:")
    inputs = keras.Input(shape=(dataFile.sequence_length,
                                dataFile.num_components))
    x = layers.Conv1D(8, 24, activation="relu")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, 12, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, 6, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "jena_conv.keras",
            save_best_only=True) ]
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        dataFile.train_dataset,
        epochs=10,
        validation_data=dataFile.val_dataset,
        callbacks=callbacks, )

    model = keras.models.load_model("jena_conv.keras")
    results = model.evaluate(dataFile.test_dataset)

    import numpy as np
    print(f"  Test RMSE: {np.sqrt(results[0]):.2f}")
    print(f"  Test MAE: {results[1]:.2f}")
    
