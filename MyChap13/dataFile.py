import numpy as np
import keras

class DataFile:
    def __init__(self):
        f = open("jena_climate_2009_2016.csv")
        data = f.read()
        lines = data.split('\n')

        header = lines[0].split(',')
        lines = lines[1:]
        f.close()
        del data ###########

        print("header", header)
        print("len(lines)=", len(lines))

        self.temperature = np.zeros((len(lines),))
        self.num_components = len(header) - 1
        raw_data = np.zeros((len(lines), self.num_components))

        for i,line in enumerate(lines):
            lineSplit = line.split(',')
            values = []
            for x in lineSplit[1:]:
                values.append(float(x))
            self.temperature[i] = values[1]
            raw_data[i, :] = values[:]
        del lines ###################

        self.num_train_samples = int(0.5 * len(raw_data))
        self.num_val_samples = int(0.25 * len(raw_data))
        self.num_test_samples = len(raw_data) - self.num_train_samples - self.num_val_samples
        print("num_train_samples:", self.num_train_samples)
        print("num_val_samples:", self.num_val_samples)
        print("num_test_samples:", self.num_test_samples)

        self.mean = raw_data[:self.num_train_samples].mean(axis=0)
        raw_data -= self.mean
        self.std = raw_data[:self.num_train_samples].std(axis=0)
        raw_data /= self.std

        sampling_rate = 6
        self.sequence_length = 120
        delay = sampling_rate * (self.sequence_length + 24 - 1)
        batch_size = 256
        
        self.train_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-delay],
            targets=self.temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=self.sequence_length,
            # shuffle=True,
            batch_size=batch_size,
            start_index=0,
            end_index=self.num_train_samples,
            format="grain",
        )
        self.val_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-delay],
            targets=self.temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=self.sequence_length,
            # shuffle=True,
            batch_size=batch_size,
            start_index=self.num_train_samples,
            end_index=self.num_train_samples + self.num_val_samples,
            format="grain",
        )
        self.test_dataset = keras.utils.timeseries_dataset_from_array(
            raw_data[:-delay],
            targets=self.temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=self.sequence_length,
            # shuffle=True,
            batch_size=batch_size,
            start_index=self.num_train_samples + self.num_val_samples,
            format="grain",
        )

        del raw_data
        
            
            
            
