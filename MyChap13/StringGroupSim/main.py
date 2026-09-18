import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

#from dataFile import DataFile
import dataFile
from LSTM import LSTM

def main():
    # dataFile = DataFile()
    dataFileIph = dataFile.DataFile_Iph()
    dataFileIph.plot()
    exit()
    
    lstm = LSTM(dataFile)
    lstm.fit()
    lstm.saveHistory()
    lstm.plotSavedHistory()
    

main()

