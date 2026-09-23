import os
os.environ["KERAS_BACKEND"] = "jax"
# os.environ["JAX_PLATFORMS"] = "cpu"

from dataFile import DataFile
# from LSTM import LSTM
# from naive import naive
# from denseLayerModel import denseLayerModel
#from conv1DModel import conv1DModel
from LSTMFunc import LSTMModel
#from simpleRNN import simpleRNNModel

def main():
    dataFile = DataFile()
    # naive(dataFile)
    # denseLayerModel(dataFile)
    # conv1DModel(dataFile)
    LSTMModel(dataFile)
    # simpleRNNModel(dataFile)
    exit()

    lstm = LSTM(dataFile)
    lstm.fit()
    lstm.saveHistory()
    lstm.plotSavedHistory()

main()

