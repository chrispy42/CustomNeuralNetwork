import pygame
import mnist_loader
import random
from neuralNet import neuralNet
import numpy as np

LAYERS = [784, 30, 10]
INTERATIONS = 2
BATCH_SIZES = 3
LEARNING_RATE = 3.0

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = neuralNet(LAYERS, displayBoolean=True)
test_data = [(x, y) for (x, y) in test_data]
print("BASE TEST: "+str(round(100*(net.testStuff(test_data)/10000), 2))+"% correct")
net.learnStuff(training_data, BATCH_SIZES, INTERATIONS, LEARNING_RATE, test_data)
print("FINAL RESULT: "+str(round(100*(net.testStuff(test_data)/10000), 2))+"% correct")