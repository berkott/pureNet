from getData import mnistData
import numpy as np
import math

class network:
    def __init__(self, hidden_layers, nodes_per_layer):
        self.hiddenLayers = hidden_layers
        self.nodes = nodes_per_layer

        self.weights = self.getRandomWeights()
        self.biases = self.getRandomBiases()

        dataClass = mnistData()
        trainD, self.trainL, testD, self.testL = dataClass.getData()
        self.trainD = self.shuffle(trainD)
        
        print(len(trainD))
        print(len(self.trainD))
        
        self.testD = self.shuffle(testD)

        self.activationResultsFromPass = []

    def getRandomWeights(self):
        weights = []

        # 784 is the number of indices in the image
        weights.append(np.random.uniform(-1, 1, self.nodes * 784))

        for i in range(self.hiddenLayers-1):
            weights.append(np.random.uniform(-1, 1, self.nodes ** 2))

        # there are 10 possible numbers that can be predicted, 0-9
        weights.append(np.random.uniform(-1, 1, self.nodes * 10))

        return weights

    def getRandomBiases(self):
        biases = []

        for i in range(self.hiddenLayers):
            biases.append(np.random.uniform(-1, 1, self.nodes))

        # there are 10 possible numbers that can be predicted, 0-9
        biases.append(np.random.uniform(-1, 1, 10))

        return biases

    def train(self):
        activationResultsTotal = []

        # print(len(self.trainD))
        # for i in range(len(self.trainD)):
            # print(i + 1)
        activationResultsTotal.append(self.forwardPass(0))

    def forwardPass(self, passNum):
        activationResults = []

        for i in range(self.hiddenLayers):
            activationResults.append(np.zeros(self.nodes))
        
        activationResults.append(np.zeros(10))

        # loop through layers
        for i in range(self.hiddenLayers + 1):
            # loop through nodes
            for j in range(len(self.biases[i])):
                # loop though and get the weight number for a node
                weightResult = 0
                # print("i: ", i, ", j: ", j)
                if (i == 0):
                    for k in range(int(len(self.weights[i])/self.nodes)):
                        weightResult += self.weights[i][k * (1 + j)] * self.trainD[passNum][k]
                else:
                    for k in range(int(len(self.weights[i])/self.nodes)):
                        weightResult += self.weights[i][k * (1 + j)] * activationResults[i-1][k]

                activationResults[i][j] = self.sigmoid(weightResult + self.biases[i][j])

        return activationResults

    def sigmoid(self, x):
        # return 1/(1 + math.exp(x))
        return 1/(1 + math.e ** -x)

    def shuffle(self, value):
        return np.random.shuffle(value)

    # testing methods
    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases