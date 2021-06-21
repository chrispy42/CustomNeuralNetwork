#this is the main nural network. While in this repository it is being used to recodnize digits
#it can be used for anything the user wants provided a network layout in the form of 
#[# of nodes, # of nodes, ..., # of nodes], training and test data, and all the other information
#to dictate how the network will learn. (see main for example)
#you will need to have numpy installed to run the program 
import numpy as np
import random
from display import displayWindow


class neuralNet(object):
    def __init__(self, networkLayout, displayBoolean = False):
        #layout of network in terms of number of nodes in each layer
        self.networkLayout = networkLayout
        self.weights = []
        self.biases = []

        #determines weather or not to display things
        self.displayBoolean = displayBoolean
        if self.displayBoolean:
            self.display = displayWindow(800)

        #initiates weights and biases with arrays for every non input layer (i=0) 
        for i in range(len(self.networkLayout)):
            if i != 0:
                #randn -> random numbers with no max or min but constant standard deviation
                self.weights.append(np.random.randn(self.networkLayout[i], self.networkLayout[i-1]))
                self.biases.append(np.random.randn(self.networkLayout[i]))
    
    #gets the output of the network in an array of activations for an array of inputs
    def getOutput(self, inputs):
        #standardize input format
        activations = inputs
        activations = np.asarray(activations)
        activations = np.reshape(activations, (self.networkLayout[0]))

        if len(inputs) == self.networkLayout[0]:
            #for each layer (except the inputs) take the dot product of the activations and the weights then add the biases
            for i in range(len(self.networkLayout)-1):
                activations = self.sigmoid(np.dot(self.weights[i], activations)+self.biases[i])
        else: return 0
        return activations

    #returns the total number of trials the network got correct in the test data provided
    def testStuff(self, testData):
        test_results = []
        numCorrect = 0
        for (x, y) in testData:
            #runs input through the network and adds the max index of the result array and the expected output
            test_results.append((np.argmax(self.getOutput(x)), y))
        #checks for correct test cases
        for (x, y) in test_results:
            if x==y:
                numCorrect+=1
        return numCorrect

    def learnStuff(self, trainingData, batchSizes, numOfItertaions, learningRate, testData):
        #does the learning, trainingData and testData are the data to train and test with (test not required), batch sizes is num
            #of test cases the network will use to train itself off of, iterations is how many times the network will work through
            #all of the training data, learning rate is how fast the gradient decsent will take place
        
        #displays initial neural network
        if self.displayBoolean:
            self.display.drawNeuralNet(self.networkLayout, self.weights, self.biases)

        trainingData = list(trainingData)
        trainingLength = len(trainingData)

        for j in range(numOfItertaions):
            #randomize order to have random batches
            random.shuffle(trainingData)

            #separate data into seperate batches (note: integer division here means not all training examples will be used 
                # unless batch size divides evenly into the number of examples)
            for i in range(int(trainingLength/batchSizes)-1):
                #use stochastic gradient decent to adjust the weights and biases of the network for each batch 
                self.learnFromBatch(trainingData[i*batchSizes:(i+1)*batchSizes], learningRate)
                
                #commentented code can be used to display results of network inbetween batchs
                #updates display every 100 batchs (updating every batch results in no visible change and wastes time)
                if (i%100 == 0):
                    if self.displayBoolean:
                        self.display.drawNeuralNet(self.networkLayout, self.weights, self.biases)
                    '''num = self.testStuff(testData)
                    print(str(round(100*(num/10000), 2))+"% correct, batch num: "+str(i))'''

            #tests data again, prints the results for that iteration, updates the display
            num = self.testStuff(testData)
            print("INTERATION "+str(j)+": "+str(round(100*(num/10000), 2))+"% correct")
            if self.displayBoolean:
                self.display.drawNeuralNet(self.networkLayout, self.weights, self.biases)

        #this code just looks at the examples the network got wrong and prints them on the screen
        if self.displayBoolean:
            incorrectAns = []
            for (x, y) in testData:
                if (np.argmax(self.getOutput(x)) != y):
                    incorrectAns.append((x, np.argmax(self.getOutput(x))))
            for i in range(min(70, len(incorrectAns))):
                if i<min(70, len(incorrectAns))-1:
                    self.display.drawExNum(incorrectAns[i][0], incorrectAns[i][1])
                else:
                    self.display.drawExNum(incorrectAns[i][0], incorrectAns[i][1], keep=True)

            
            

    def learnFromBatch(self, batch, learningRate):
        #reinitializes weight and biases vectors that will eventually be added to real weights and biases
        weightVectors = []
        biaseVectors = []
        for i in range(len(self.networkLayout)):
            if i != 0:
                #0's instead of random because the vectors will be added at the end
                weightVectors.append(np.zeros([self.networkLayout[i], self.networkLayout[i-1]]))
                biaseVectors.append(np.zeros(self.networkLayout[i]))
        
        #for ever trial in a batch
        for ex in batch:
            #do the backpropogation calculus to get the vectors that best improve the netowrk for that example
            save = self.calcBackpropogationVectors(ex)
            exweightVector = save[0]
            exbiaseVector = save[1]

            #add all of the vectors from the batch together and later divde it by the number of trials in the batch to 
                #create an average of every backpropgation vector in the batch (the vectors to best improve the network overall)
            weightVectors = [wv+exwv for wv, exwv in zip(weightVectors, exweightVector)]
            biaseVectors = [bv+exbv for bv, exbv in zip(biaseVectors, exbiaseVector)]
        #adds the vectors multiplied by the learning rate to the actual weights and biases of the network 
            #(neg learning rate bc gradient descent gives slope of increase so neg decrases the cost function)
        weightVectors = [(-learningRate)*(wv/len(batch)) for wv in weightVectors]
        biaseVectors = [(-learningRate)*(bv/len(batch)) for bv in biaseVectors]
        self.weights = [wv+sw for wv, sw in zip(weightVectors, self.weights)]
        self.biases = [bv+sb for bv, sb in zip(biaseVectors, self.biases)]

    def calcBackpropogationVectors(self, ex):
        #calculates the vectors to optomize the cost function for a given example
        exin, exout = ex
        exweightVectors = []
        exbiaseVectors = []
        
        #initialize the vectors as arrays of 0s 
        for i in range(len(self.networkLayout)):
            if i != 0:
                exweightVectors.append(np.zeros([self.networkLayout[i], self.networkLayout[i-1]]))
                exbiaseVectors.append(np.zeros(self.networkLayout[i]))
        
        #compute the activations and Zs (z is the activation before it is put into the sigmoid function)
        totalActivations = []
        totalZs = []
        #reshape the input to a universal format
        exin = np.reshape(exin, (self.networkLayout[0]))
        for i in range(len(self.networkLayout)-1):
            if i == 0:
                #for the first layer the input is used
                totalZs.append(np.dot(self.weights[i], exin)+self.biases[i])
                totalActivations.append(self.sigmoid(totalZs[-1]))
            else:
                #for every other layer the previous activations are used (totactiv[-1])
                totalZs.append(np.dot(self.weights[i], totalActivations[-1])+self.biases[i])
                totalActivations.append(self.sigmoid(totalZs[-1]))
        
        #compute output Error for layers
        #all of the errors here are computed with a cost function of 1/2(a-e)^2. The cost function is never calculated
        #because only the derivative (a-e) is needed to compute the gradient of the function.
        errorVectors = []
        costDerivative = []
        exout = np.asarray(exout)

        #for every output compute the difference between expected and actual
        for i in range(len(totalActivations[-1])):
            costDerivative.append(totalActivations[-1][i]-exout[i])
        #reshape 
        costDerivative = np.asarray(costDerivative)
        costDerivative = np.reshape(costDerivative, (self.networkLayout[-1]))
        #this equation gives the error of the last layer of the network
        errorVectors.append((costDerivative)*(self.sigmoidPrime(totalZs[-1])))


        #compute error for every layer with backpropogation calculus
        #this calculation is similar to the one above. The backpropogation comes from using the next layers error 
        #in the calculation for that layers error
        for i in range(1, len(self.networkLayout)-1):
            errorVectors.insert(0, np.dot(errorVectors[0], self.weights[-i])*(self.sigmoidPrime(totalZs[-(i+1)])))
        #reshape
        for i in range(len(errorVectors)):
            errorVectors[i] = np.reshape(errorVectors[i], [len(errorVectors[i]), 1])
            totalActivations[i] = np.reshape(totalActivations[i], [len(totalActivations[i]), 1])
            exin = np.reshape(exin, [len(exin), 1])
        #the actual weight vectors are now calculated as the dot product of the activations of the prev layer and
            #the error vectors of that layer. The biase vectors are just the errors
        for i in range(len(errorVectors)):
            #use the activations for everything but the first layer, use the input on the first layer
            if (i+2<=len(totalActivations)):
                exweightVectors[-i-1] = np.dot(errorVectors[-i-1], np.transpose(totalActivations[(-i-2)]))
            else:
                exweightVectors[-i-1] = np.dot(errorVectors[-i-1], np.transpose(exin))
            #biase vector is done here
            exbiaseVectors[i] = [x[0] for x in errorVectors[i]]
        #return the weight and biase vectors for that example
        return (exweightVectors, exbiaseVectors)           
    
    
    def sigmoid(self, x):
        #logistic function to fit output between 0 and 1 (sigmoid here)
        return 1.0/(1.0+np.exp(-x))
    
    def sigmoidPrime(self, x):
        #the derivative of the sigmoid fnction used for the backpropogation
        return self.sigmoid(x)*(1-self.sigmoid(x))