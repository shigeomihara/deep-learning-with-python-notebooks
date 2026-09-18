############################ main.py ##################################
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
        # loss=["mean_squared_error", "sparse_categorical_crossentropy"],
        # metrics=[["mean_absolute_error"], ["accuracy"]],
        loss=["sparse_categorical_crossentropy"],
        # loss=["mean_squared_error"],
        metrics=["accuracy"],
        # metrics=["mean_absolute_error"],
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
        train_images,
        train_labels,
        epochs=10,
        callbacks=callbacks_list,
        validation_data=(val_images, val_labels),
    )

main()

