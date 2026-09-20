import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras

int_sequence = np.arange(10)

dummy_data = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],
    targets=int_sequence[3:],
    sequence_length=3,
    shuffle=True,
    batch_size=2,
    format="grain"
    )

for inputs, targets in dummy_data:
    print(inputs, targets)
    
