from jax import numpy as jnp
import jax

input_dim=2
output_dim=1

def model(inputs, W, b):
    return jnp.matmul(inputs, W)+b

inputs=jnp.array([[2,3],
                  [-1,4]])
targets=jnp.array([[5],
                   [-3]])

print("inputs=",inputs,"\ntargets=", targets)

import numpy as np

W = jnp.array(np.random.uniform(size=(input_dim, output_dim)))
b = jnp.array(np.random.uniform(size=(output_dim, )))

print("W=", W)
print("b=", b)

predictions = model(inputs, W, b)
print("predictions=", predictions)

def mean_square_error(targets, predictions):
    per_sample_losses = jnp.square(targets-predictions)
    return jnp.mean(per_sample_losses)

print("meas_square_error=", mean_square_error(targets, predictions))

def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_square_error(targets, predictions)
    return loss

loss = compute_loss((W,b), inputs, targets)
print("loss=", loss)

grad_fn = jax.value_and_grad(compute_loss)

learning_rate = 0.01

@jax.jit
def training_step(inputs, targets, W, b):
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    W = W - grad_wrt_W*learning_rate
    b = b - grad_wrt_b*learning_rate
    return loss, W, b

for step in range(40):
    loss, W, b = training_step(inputs, targets, W, b)
    print(f"loss at step {step}: {loss:.4f}")
    
    



