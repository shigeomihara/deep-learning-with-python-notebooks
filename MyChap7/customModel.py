############################ customModel.py ##########################
import jax
import keras
from keras import layers
from keras.src import backend ####################

class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layer = layers.Dense(512, activation="relu")
        self.dropout_layer = layers.Dropout(0.5)
        self.output_layer = layers.Dense(10, activation="softmax")

    def call(
            self,
            inputs,
            **kwargs,######################
    ):
        features = self.dense_layer(inputs)
        features = self.dropout_layer(features)
        outputs = self.output_layer(features)
        return outputs
    
    # def compute_loss_and_updates(
    #     self,
    #     trainable_variables,
    #     non_trainable_variables,
    #     metrics_variables, ##############
    #     inputs,
    #     targets,
    #     sample_weight, ###########################
    #     training=False,
    #     optimizer_variables=None,#######################
    # ):
    #     """This method is stateless and is intended for use with jax.grad."""
    #     kwargs = {}
    #     if self._call_has_training_arg:
    #         kwargs["training"] = training

    #     predictions, non_trainable_variables, losses = self.stateless_call(
    #         trainable_variables,
    #         non_trainable_variables,
    #         inputs,
    #         return_losses=True,############
    #         # training=training,##############
    #         **kwargs,####################
    #     )
        
    #     if losses:
    #         # Make forward pass losses available to compute_loss.
    #         self._losses_override.clear()
    #         self._losses_override = losses

    #     loss, variables = self.stateless_compute_loss(
    #         trainable_variables,
    #         non_trainable_variables,
    #         metrics_variables,
    #         # x=x,
    #         # y=y,
    #         # y_pred=y_pred,
    #         x=inputs,
    #         y=targets,
    #         y_pred=predictions,
    #         sample_weight=sample_weight,
    #         training=training,
    #     )
    #     if losses:
    #         self._losses_override.clear()
    #     (trainable_variables, non_trainable_variables, metrics_variables) = (
    #         variables
    #     )

    #     # Handle loss scaling
    #     unscaled_loss = loss
    #     if training and self.optimizer is not None:
    #         # Scale loss with a StatelessScope, to use an update scale variable.
    #         mapping = list(zip(self.optimizer.variables, optimizer_variables))
    #         with backend.StatelessScope(state_mapping=mapping):
    #             loss = self.optimizer.scale_loss(loss)
    #     return loss, (
    #         unscaled_loss,
    #         #y_pred,
    #         predictions,
    #         non_trainable_variables,
    #         metrics_variables,
    #     )
    #     # loss = self.compute_loss(y=targets, y_pred=predictions)
    #     # return loss, (predictions, non_trainable_variables)
    #     # return losses, (predictions, non_trainable_variables)

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

        #(loss, (predictions, non_trainable_variables)), grads = grad_fn(
        (loss, (unscaled_loss, predictions, non_trainable_variables,
                metrics_variables)), grads = grad_fn(
        # (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,###############
            inputs,
            targets,
            sample_weight=None,
            training=True,
            optimizer_variables=optimizer_variables,
        )
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_metrics_vars = []
        logs = {}
        for metric in self.metrics:
            num_prev = len(new_metrics_vars)
            num_current = len(metric.variables)
            current_vars = metrics_variables[num_prev : num_prev + num_current]
            if metric.name == "loss":
                current_vars = metric.stateless_update_state(current_vars, loss)
            else:
                current_vars = metric.stateless_update_state(
                    current_vars, targets, predictions
                )
            logs[metric.name] = metric.stateless_result(current_vars)
            new_metrics_vars += current_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state    

# from keras import layers

# def get_custom_model():
#     inputs = keras.Input(shape=(28 * 28,))
#     features = layers.Dense(512, activation="relu")(inputs)
#     features = layers.Dropout(0.5)(features)
#     outputs = layers.Dense(10, activation="softmax")(features)
#     model = CustomModel(inputs, outputs)
#     model.compile(optimizer=keras.optimizers.Adam())
#     return model

