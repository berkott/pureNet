from getData import mnistData
from network import network

# Getting Data
dataClass = mnistData()
trD, trL, teD, teL = dataClass.getData()

# Neural Net

neuralNet = network(2, 16)
neuralNet.train()
