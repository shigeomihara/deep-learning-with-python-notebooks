############################ main.py ##################################
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import keras
import mnistData
import customModel

def main():
    ((train_images, train_labels),
     (val_images, val_labels),
     (test_images, test_labels)) = mnistData.getData()

    model = customModel.CustomModel()
    
    model.compile(
        optimizer="adam",
        #loss=["mean_square_error", "sparse_categorical_crossentropy"],
        loss=["sparse_categorical_crossentropy"],
        # metrics=[["mean_absolute_error"], ["accuracy"]],
        metrics=["accuracy"],
        )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="accuracy",
            patience=1,
            ),
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoint.keras",
            monitor="val_loss",
            save_best_only=True,
            ),
        ]
        
    model.fit(
        x=train_images,
        y=train_labels,
        epochs=3,
        callbacks=callbacks_list,
        validation_data=(val_images, val_labels),
    )

main()

