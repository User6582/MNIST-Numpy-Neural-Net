import numpy as np
import idx2numpy as idx2np
import pickle
from matplotlib import pyplot as plt
from DisplayImage import display_image
saveFreq = 1000

class NeuralNetwork:
    def __init__(self, setting = None):
        self.input = None
        if setting is None:
            weights1 = 2*np.random.rand(784, 16)-1
            biases1 = np.ones(16)
            weights2 = 2 * np.random.rand(16, 16) - 1
            biases2 = np.ones(16)
            weights3 = 2 * np.random.rand(16, 10)-1
            biases3 = np.ones(10)
        else:
            weights1 = setting[0]
            biases1 = setting[1]
            weights2 = setting[2]
            biases2 = setting[3]
            weights3 = setting[4]
            biases3 = setting[5]
        self.weights1 = weights1
        self.biases1 = biases1
        self.weights2 = weights2
        self.biases2 = biases2
        self.weights3 = weights3
        self.biases3 = biases3
        self.output = np.zeros(10)
        self.layer1 = []
        self.layer2 = []
        self.settings = [self.weights1, self.biases1, self.weights2, self.biases2, self.weights3, self.biases3]
        self.sumCorrect = 0
        self.gradients = [ ]
        self.expected = []
        self.nSettings = []
        self.sumCost = 0
        self.learningRate = 1
        self.avgCost = []
        self.avgCorrectness = []
        self.saveNum = 0

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1)+self.biases1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.biases2)
        self.output = sigmoid(np.dot(self.layer2, self.weights3)+self.biases3)

    def get_output(self, input1, label, test=False, saveScores=False):
        input1 = input1.flatten()
        self.input = input1/255
        self.feed_forward()
        if not test:
            self.sumCost += self.get_cost(label)
            self.full_back_prop()
        if saveScores:
            self.save()
        else:
            self.saveNum += 1
        result = np.where(self.output == np.amax(self.output))[0]
        return result

    def save(self):
        self.avgCorrectness.append(self.sumCorrect / self.saveNum)
        self.avgCost.append(self.sumCost / self.saveNum)
        self.saveNum = 0
        self.sumCorrect = 0
        self.sumCost = 0
        print("correct:", self.avgCorrectness[-1])
        #print("cost:", self.avgCost[-1])

    def get_cost(self, label):
        self.expected = np.zeros(10)
        self.expected[label] = 1
        costs = (self.output-self.expected)**2
        sumCosts = costs.sum()
        return sumCosts

    def back_prop_layer(self, layerW , layerB , prevLayerA, doDa=False):
        z = sigmoid(np.dot(prevLayerA, layerW)+layerB)
        DaDz = z*(1-z)
        DaDw = DaDz*prevLayerA[:, np.newaxis]
        DaDb = np.array(DaDz)
        if doDa:
            DaDa1 = DaDz[np.newaxis, :]* layerW
            return DaDw, DaDb, DaDa1
        else:
            return DaDw, DaDb

    def full_back_prop(self):
        Da1Dw1, Da1Db1 = self.back_prop_layer(self.weights1,self.biases1,self.input)
        Da2Dw2, Da2Db2, Da2Da1 = self.back_prop_layer(self.weights2,self.biases2,self.layer1, doDa=True)
        Da3Dw3, Da3Db3, Da3Da2 = self.back_prop_layer(self.weights3,self.biases3,self.layer2, doDa=True)
        DcDa3 = 2*(self.output-self.expected)
        DcDw3 = Da3Dw3 * DcDa3
        DcDb3 = Da3Db3 * DcDa3
        DcDa2 = (Da3Da2 * DcDa3).sum(axis=1)
        DcDw2 = Da2Dw2 * DcDa2
        DcDb2 = Da2Db2 * DcDa2
        DcDa1 = (Da2Da1 * DcDa2).sum(axis=1)
        DcDw1 = Da1Dw1 * DcDa1
        DcDb1 = Da1Db1 * DcDa1
        nSetting = [DcDw1,DcDb1,DcDw2,DcDb2,DcDw3,DcDb3]
        self.nSettings.append(nSetting)

    def update(self):
        for i in range(len(self.settings)):
            change = np.zeros(self.settings[i].shape)
            for nSetting in self.nSettings:
                change = change - np.array(nSetting[i])
            change = change/len(self.nSettings)
            self.settings[i] += change * self.learningRate
        self.nSettings = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_NN(num):
    with open("save{0}.txt".format(num), "rb")  as f:
        p = pickle.load(f)
    s = p[0]
    NN = NeuralNetwork(setting=s)
    return NN


def save(NN,num):
    with open("save{0}.txt".format(num), "wb") as f:
        pickle.dump([NN.settings, [NN.avgCorrectness, NN.avgCost]], f)


def get_training(testFile,labelFile):
    tests = idx2np.convert_from_file("train-images.idx3-ubyte")
    labels = idx2np.convert_from_file("train-labels.idx1-ubyte")
    indices = np.arange(np.shape(tests)[0])
    np.random.shuffle(indices)
    tests = tests[indices]
    labels = labels[indices]
    labels = np.reshape(labels, (600,100) )
    tests = np.reshape(tests, (600,100,28,28) )
    return tests, labels


def get_tests():
    tests = idx2np.convert_from_file("train-images.idx3-ubyte")
    labels = idx2np.convert_from_file("train-labels.idx1-ubyte")
    return tests, labels


def trainNN():
    for i in range(49,60):
        NN = get_NN(i)
        tests, labels = get_training("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        zipped = zip(tests, labels)
        t = 0
        for groupTest, groupLabel in zipped:
            for test, label in zip(groupTest,groupLabel):
                t+=1
                saveScores = False
                if t % saveFreq == 0:
                    saveScores = True
                output = NN.get_output(test, label, saveScores=saveScores)
                if output == label:
                    NN.sumCorrect += 1
            NN.update()
        save(NN, i)


def testNN():
    NN = get_NN(50)
    tests, labels = get_tests()
    t = 0
    for i in range(len(tests)):
        test = tests[i]
        label = labels[i]
        output = NN.get_output(test, label, test=True)
        t += 1
        if output == label:
            NN.sumCorrect += 1
        else:
            print("output:", output[0])
            display_image(test,label)
    print("total correct:",NN.sumCorrect)
    print("total wrong:",len(tests)-NN.sumCorrect)

#testNN()
trainNN()