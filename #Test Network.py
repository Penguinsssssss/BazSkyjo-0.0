#NN test page

import random
import math
import numpy as np #type: ignore
np.random.seed(0)

e = math.e

# activation functions
def sig(input): return 1 / (1 + np.exp(-input)) # sigmoid function
def relu(input): return np.maximum(0, input) # rectified linear
def expo(input): return np.exp(input) # exponent

class Network:
    
    def __init__(self, layerstruct, num_actions):
        
        # init all layers
        self.layers = []
        for i in range(len(layerstruct) - 1):
            self.layers.append(Layer(layerstruct[i], layerstruct[i + 1]))
        
        self.value = Layer(layerstruct[-1], 1) # value layer
        self.advantage = Layer(layerstruct[-1], num_actions) # advantage head
    
    def calculate(self, input):
        
        #pass input through layers
        output = None
        for i in range(len(self.layers)): output = self.layers[i].calculate(input if i == 0 else output) # hidden layers
        
        v = self.value.calculate(output) # value
        a = self.advantage.calculate(output) # advantage
        a_mean = np.mean(a, axis=1, keepdims=True)
        q = v + (a - a_mean)
        
        return q

class Layer:
    
    def __init__(self, numinputs, numneurons):
        
        # init starting values (currently random)
        self.weights = 0.1 * np.random.randn(numinputs, numneurons)
        self.biases = np.zeros([1, numneurons])
        
        # currently using relu, will look into gelu or tanh
        self.function = relu
        
        # assume relu unless specified otherwise
        self.boollast = False
    
    def calculate(self, inputs):
        base = np.dot(inputs, self.weights) + self.biases # initial values
        activated = self.function(base) # activation function
        return activated

class DuelingDQN:
    def __init__(self):
        pass

class ReplayBuffer:
    def __init__(self):
        pass

def main():
    bot = Network([24, 2048, 2048, 2048, 2048]) # input neurons per layer
    
    data = []
    for _ in range(10): # no. of samples
        tdata = []
        for _ in range(24): tdata.append(random.random()) # number of values per sample
        data.append(tdata)
    
    print(f"OUTPUT: \n {bot.calculate(np.array([[0.1]*24]))}")

if __name__ == "__main__":
    main()