import numpy as np
import keras
import matplot

class DataFile_Iph:
    def __init__(self):
        f = open("GTIVTimeSim_201905_Iph.dat")
        data = f.read()
        lines = data.split('\n')

        header = lines[0].split(',')
        if len(lines[-1]) < 3:
            lines = lines[1:-1]
            print("last empty line")
        else:
            lines = lines[1:]            
        f.close()

        print("header", header)
        print("len(lines)=", len(lines))

        self.time_str = []
        self.zeroOnes = np.zeros((len(lines),))
        
# header ['Time', ' Gmeas', ' Tmeas', ' Imeas', ' Vmeas', ' ImpSim', ' VmpSim', ' IphSim', ' 0or1']
# Time, Imeas, Vmeas, IphSim を除外
        self.raw_data = np.zeros((len(lines), len(header)-4)) 

        for i,line in enumerate(lines):
            lineSplit = line.split(',')
            values = []
            for x in lineSplit[1],lineSplit[2],lineSplit[5],lineSplit[6],lineSplit[8]:
                values.append(float(x))
                
            self.time_str.append(lineSplit[0])  # Time
            self.zeroOnes[i] = float(lineSplit[8])  # 0or1
            self.raw_data[i, :] = values[:] # Gmeas, Tmeas, ImpSim, VmpSim, 0or1


    def plot(self):
        matplot.plot(self.time_str, self.raw_data[:,2])
            

class DataFile:
    def __init__(self):
#        f = open("GTIVTimeSim_201905_Rs_Mul100.dat") 
        f = open("GTIVTimeSim_201905_Iph.dat")
        data = f.read()
        lines = data.split('\n')

        header = lines[0].split(',')
        lines = lines[1:]
        f.close()

        print("header", header)
        print("len(lines)=", len(lines))

        self.RsSim = np.zeros((len(lines),))
        
        # Time, Imeas, Vmeas, Rs, RsSim, mul を除外
        self.raw_data = np.zeros((len(lines), len(header)-6)) 

        for i,line in enumerate(lines):
            lineSplit = line.split(',')
            values = []
            for x in lineSplit[1],lineSplit[2],lineSplit[5],lineSplit[6]:
                values.append(float(x))
                
            self.RsSim[i] = lineSplit[8]  # RsSim
            self.raw_data[i, :] = values[:]

        print(self.raw_data)
        exit()

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

        sampling_rate = 6
        sequence_length = 120
        delay = sampling_rate * (sequence_length + 24 - 1)
        batch_size = 256
        
        self.train_dataset = keras.utils.timeseries_dataset_from_array(
            self.raw_data[:-delay],
            targets=self.temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=True,
            batch_size=batch_size,
            start_index=0,
            end_index=self.num_train_samples,
        )
        self.val_dataset = keras.utils.timeseries_dataset_from_array(
            self.raw_data[:-delay],
            targets=self.temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=True,
            batch_size=batch_size,
            start_index=self.num_train_samples,
            end_index=self.num_train_samples + self.num_val_samples,
        )
        self.test_dataset = keras.utils.timeseries_dataset_from_array(
            self.raw_data[:-delay],
            targets=self.temperature[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=True,
            batch_size=batch_size,
            start_index=self.num_train_samples + self.num_val_samples,
        )

            
            
            
