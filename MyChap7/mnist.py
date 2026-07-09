import os
os.environ["KERAS_BACKEND"] = "jax"

from keras.datasets import mnist
from keras import layers
import keras

def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

(images, labels), (test_images, test_labels) = mnist.load_data()

images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

print(train_images[15:20])
exit()

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# def compute_loss_and_updates(
#         trainable_variables, non_trainable_variables, inputs, targets
# ):
#     outputs, non_trainable_variables = model.stateless_call(
#         trainable_variables, non_trainable_variables, inputs, training=True
#     )
#     loss = loss_fn(targets, outputs)
#     return loss, non_trainable_variables

import jax
# grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

optimizer = keras.optimizers.Adam()
optimizer.build(model.trainable_variables)

# def train_step(state, inputs, targets):
#     (trainable_variables, non_trainable_variables, optimizer_variables) = state
#     (loss, non_trainable_variables), grads = grad_fn(
#         trainable_variables, non_trainable_variables, inputs, targets
#     )
#     trainable_variables, optimizer_variables = optimizer.stateless_apply(
#         optimizer_variables, grads, trainable_variables
#     )
#     return loss, (
#         trainable_variables,
#         non_trainable_variables,
#         optimizer_variables,
#     )

batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]

trainable_variables = [v.value for v in model.trainable_variables]
non_trainable_variables = [v.value for v in model.non_trainable_variables]
optimizer_variables = [v.value for v in optimizer.variables]

# state = (trainable_variables, non_trainable_variables, optimizer_variables)
# loss, state = train_step(state, inputs, targets)

loss_fn = keras.losses.SparseCategoricalCrossentropy()

class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        targets,
        training=False,
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            inputs,
            training=training,
        )
        loss = loss_fn(targets, predictions)
        return loss, non_trainable_variables

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        inputs, targets = data

        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            targets,
            training=True,
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        logs = {"loss": loss}
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        return logs, state

def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam())
    return model

model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
