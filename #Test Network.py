#NN test page

import random
import math
import numpy as np
np.random.seed(0)

e = math.e

#activation functions
def linear(input): return input
def sig(input): return 1 / (1 + np.exp(-input))
def relu(input): return np.maximum(0, input)
def expo(input): return np.exp(input)

class Network:
    
    def __init__(self, layerstruct):
        
        #init all layers
        self.layers = []
        for i in range(len(layerstruct) - 1):
            self.layers.append(Layer(layerstruct[i], layerstruct[i + 1]))
        
        #set last layer to use expo, not relu
        self.layers[-1].boollast = True
    
    def calculate(self, input):
        
        #pass input through layers
        output = None
        for i in range(len(self.layers)): output = self.layers[i].calculate(input if i == 0 else output)
        return output

class Layer:
    
    def __init__(self, numinputs, numneurons):
        
        #init starting values (currently random)
        self.weights = 0.1 * np.random.randn(numinputs, numneurons)
        self.biases = np.zeros([1, numneurons])
        
        #currently using relu, will look into gelu or tanh
        self.function = relu
        
        #assume relu unless specified otherwise
        self.boollast = False
    
    def calculate(self, inputs):
        base = np.dot(inputs, self.weights) + self.biases #initial values
        if not self.boollast: return self.function(base) #activation function
        else:
            exp_values = np.exp(base - np.max(base, axis=1, keepdims=True)) #use expo instead of relu on last layer
            return exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalize values

def main():
    bot = Network([24, 2048, 2048, 2048, 2048, 2]) #input neurons per layer
    
    data = []
    for _ in range(10): #no. of samples
        tdata = []
        for _ in range(24): tdata.append(random.random()) #number of values per sample
        data.append(tdata)
    
    print(f"OUTPUT: \n {bot.calculate(np.array([[0.1]*24]))}")

if __name__ == "__main__":
    main()