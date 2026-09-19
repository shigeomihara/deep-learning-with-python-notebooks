############################ customModel.py ##########################
import tensorflow as tf
import keras
from keras import layers

loss_fn = keras.losses.SparseCategoricalCrossentropy()
#loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layer = layers.Dense(512, activation="relu")
        self.dropout_layer = layers.Dropout(0.5)
        self.output_layer = layers.Dense(10, activation="softmax")

    def call(
            self,
            inputs,
    ):
        features = self.dense_layer(inputs)
        features = self.dropout_layer(features)
        outputs = self.output_layer(features)
        return outputs
    
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}
        
    # @property
    # def metrics(self):
    #     return [loss_tracker]
    
