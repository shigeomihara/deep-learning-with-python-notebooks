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
        
    model.fit(
        train_images,
        train_labels,
        epochs=3,
        validation_data=(val_images, val_labels),
    )

main()

