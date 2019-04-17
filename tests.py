from network import network
from getData import mnistData
import unittest

class unitTests(unittest.TestCase):
    def setUp(self):
        self.testLayers = 2
        self.testNodes = 16
        self.neuralNet = network(self.testLayers, self.testNodes)

    def testNetworkDim(self):
        # Expected value for the Netwok Dim
        expectedTotal = 784 * self.testNodes

        for i in range(self.testLayers-1):
            expectedTotal += self.testNodes ** 2
        
        expectedTotal += 10 * self.testNodes

        expectedTotal += self.testNodes * self.testLayers + 10

        # Actual value for the Netwok Dim
        weights = self.neuralNet.getWeights()
        biases = self.neuralNet.getBiases()

        actualTotal = 0
        for i in range(len(weights)):
            actualTotal += len(weights[i])

        for i in range(len(biases)):
            actualTotal += len(biases[i])

        self.assertEqual(expectedTotal, actualTotal)

    def testData(self):
        dataClass = mnistData()
        trD, trL, teD, teL = dataClass.getData()
        self.assertEqual(784, len(trD[0]))
        self.assertEqual(5, trL[0])
        self.assertEqual(784, len(teD[0]))
        self.assertEqual(7, teL[0])

if __name__ == '__main__':
    unittest.main()