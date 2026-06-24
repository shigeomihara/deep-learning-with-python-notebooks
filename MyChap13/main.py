# import os
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["JAX_PLATFORMS"] = "cpu"

from dataFile import DataFile
from LSTM import LSTM

def main():
    dataFile = DataFile()
    lstm = LSTM(dataFile)
    lstm.fit()
    lstm.saveHistory()
    lstm.plotSavedHistory()
    

main()

