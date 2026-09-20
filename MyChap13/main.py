import os
os.environ["KERAS_BACKEND"] = "jax"
# os.environ["JAX_PLATFORMS"] = "cpu"

from dataFile import DataFile
from LSTM import LSTM
from naive import naive
from denseLayerModel import denseLayerModel

def main():
    dataFile = DataFile()
    # naive(dataFile)
    denseLayerModel(dataFile)
    exit()

    lstm = LSTM(dataFile)
    lstm.fit()
    lstm.saveHistory()
    lstm.plotSavedHistory()
    

main()

