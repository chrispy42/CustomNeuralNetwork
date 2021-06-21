#this is the main class where the network is run. This can be run in a python shell but it is done here
#for ease of use. All of the constant variables can be changed to adjust the preformance of the network 
#you will need the neuralNet class, the mnist_loader class, and the mnist data set for this to work

import mnist_loader
from neuralNet import neuralNet

#each index in the layers function represents the number of nodes in that layer. For this example to
#work the input must be 784 and the output must be 10 to match the data set but the hidden layers can
#be whatever the user would like.
LAYERS = [784, 30, 10]
#this is how many times the network will run through the training data and learn
INTERATIONS = 5
#this is how many examples will be used in a single batch to adjust the weights and biases of the network
BATCH_SIZES = 10
#this is how fast the network will learn. (increasing this will decrease preformance due to gapping local minimums)
LEARNING_RATE = 3.0

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = neuralNet(LAYERS, displayBoolean=True)
test_data = [(x, y) for (x, y) in test_data]
print("BASE TEST: "+str(round(100*(net.testStuff(test_data)/10000), 2))+"% correct")
net.learnStuff(training_data, BATCH_SIZES, INTERATIONS, LEARNING_RATE, test_data)
print("FINAL RESULT: "+str(round(100*(net.testStuff(test_data)/10000), 2))+"% correct")