############################ main.py ##################################
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import mnistData
import customModel

def main():
    ((train_images, train_labels),
     (val_images, val_labels),
     (test_images, test_labels)) = mnistData.getData()

    model = customModel.get_custom_model()
    model.fit(train_images, train_labels, epochs=3)

main()

