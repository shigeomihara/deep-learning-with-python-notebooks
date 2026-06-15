import numpy as np

class DataFile:
    def __init__(self):
        f = open("jena_climate_2009_2016.csv")
        data = f.read()
        lines = data.split('\n')

        header = lines[0].split(',')
        lines = lines[1:]
        f.close()

        print("header", header)
        print("len(lines)=", len(lines))

        self.temperature = np.zeros((len(lines),))
        self.raw_data = np.zeros((len(lines), len(header) - 1))

        for i,line in enumerate(lines):
            lineSplit = line.split(',')
            values = []
            for x in lineSplit[1:]:
                values.append(float(x))
            self.temperature[i] = values[1]
            self.raw_data[i, :] = values[:]

        self.num_train_samples = int(0.5 * len(self.raw_data))
        self.num_val_samples = int(0.25 * len(self.raw_data))
        self.num_test_samples = len(self.raw_data) - self.num_train_samples - self.num_val_samples
        print("num_train_samples:", self.num_train_samples)
        print("num_val_samples:", self.num_val_samples)
        print("num_test_samples:", self.num_test_samples)

        mean = self.raw_data[:self.num_train_samples].mean(axis=0)
        self.raw_data -= mean
        std = self.raw_data[:self.num_train_samples].std(axis=0)
        self.raw_data /= std

        # print(self.raw_data)
        
            
            
            
