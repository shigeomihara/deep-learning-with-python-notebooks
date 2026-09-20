import keras
from keras import layers

def denseLayerModel(dataFile):
    print("denseLayerModel:")
    inputs = keras.Input(shape=(dataFile.sequence_length,
                                dataFile.num_components))
    x = layers.Flatten()(inputs)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "jena_dense.keras",
            save_best_only=True) ]
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    # history = model.fit(
    #     dataFile.train_dataset,
    #     epochs=10,
    #     validation_data=dataFile.val_dataset,
    #     callbacks=callbacks, )

    model = keras.models.load_model("jena_dense.keras")
    results = model.evaluate(dataFile.test_dataset)

    import numpy as np
    print(f"  Test RMSE: {np.sqrt(results[0]):.2f}")
    print(f"  Test MAE: {results[1]:.2f}")
    
