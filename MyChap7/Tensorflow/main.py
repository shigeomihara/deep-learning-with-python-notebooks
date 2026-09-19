############################ main.py ##################################
import keras
import mnistData
import customModel
from rootMeanSquaredError import RootMeanSquaredError

def main():
    ((train_images, train_labels),
     (val_images, val_labels),
     (test_images, test_labels)) = mnistData.getData()

    model = customModel.CustomModel()

    model.compile(
        optimizer="adam",
        # loss=["mean_squared_error", "sparse_categorical_crossentropy"],
        # metrics=[["mean_absolute_error"], ["accuracy"]],
        loss=["sparse_categorical_crossentropy"],
        # loss=["mean_squared_error"],
        # metrics=["accuracy"],
        metrics=["accuracy", RootMeanSquaredError()],
        # metrics=["mean_absolute_error"],
        )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            # monitor="accuracy",
            monitor="loss",
            patience=1,
            ),
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoint.keras",
            monitor="val_loss",
            save_best_only=True,
            ),
        ]
        
    model.fit(
        train_images,
        train_labels,
        epochs=3,
        batch_size=128,
        callbacks=callbacks_list,
        validation_data=(val_images, val_labels),
    )

    test_results = model.evaluate(
        test_images,
        test_labels,
        )

    print(test_results)

main()

