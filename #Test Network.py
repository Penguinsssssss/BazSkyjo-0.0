#NN test page

import random
import math
import numpy as np #type: ignore
np.random.seed(0)
import copy
from collections import deque

e = math.e

# activation functions
def sig(input): return 1 / (1 + np.exp(-input)) # sigmoid function
def relu(input): return np.maximum(0, input) # rectified linear
def expo(input): return np.exp(input) # exponent

# derivative activaiton functions
def d_relu(x): return (x > 0).astype(float)

class DuelingDQN:
    
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, tau=0.01):
        
        # consts/globals
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon # exploration chance
        self.epsilon_decay = epsilon_decay # exploration rate of change
        self.epsilon_min = epsilon_min # NN will always have a small chance to explore
        self.lr = lr # learning rate of the model
        self.tau = tau # rate of main network moving to target network
        
        # replaybuffer
        self.replayBuffer = ReplayBuffer(50000) # capacity
        
        # main network
        self.model = Network([state_size, 256, 256, 256], action_size)
        
        # target network
        self.target_model = copy.deepcopy(self.model)
    
    def choose_action(self, state): # epsilon greedy function
        
        # "exploration", allows the model to sometimes choose a fully random action to get itself out of local minima
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # "explotation", the model chooses the action it thinks is best and gets the results
        q_values = self.model.calculate(state)
        return int(np.argmax(q_values))
    
    def compute_targets(self, batch): # Bellman equation
        
        # unpack a batch from replay buffer
        states, actions, rewards, next_states, dones = batch
        
        # predict Q values from main NN
        q_vals = self.model.calculate(states)
        
        #Q values from target network for next state
        next_q_vals = self.target_model.calculate(next_states)
        max_next_q = np.max(next_q_vals, axis=1)
        
        #target Q for each sample
        targets = q_vals.copy()
        
        for i in range(len(states)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]
                
        return targets
    
    def train_step(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return # prevent NN from trying to take more samples then are available
        
        # retrieve a batch from the replaybuffer
        batch = self.replaybuffer.sample(batch_size)
        
        # unpack batch
        states, actions, rewards, next_states, dones = batch
        
        # generate training targets using bellman equation
        targets = self.compute_targets(batch)
        
        # run bp on network
        self.model.backprop(states, targets, lr=self.lr)
        
        # reduce epsilon (exploration)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # use soft updates to avoid unstability
        self.soft_update()
        
    def soft_update(self): # Polyak Averaging update
        
        # hidden layers
        for layer, target_layer in zip(self.model.layers, self.target_model.layers):
            target_layer.weights = (1 - self.tau) * target_layer.weights + self.tau * layer.weights
            target_layer.biases  = (1 - self.tau) * target_layer.biases  + self.tau * layer.biases
        
        # value layer
        target_layer_v = self.target_model.value
        layer_v = self.model.value
        target_layer_v.weights = (1 - self.tau) * target_layer_v.weights + self.tau * layer_v.weights
        target_layer_v.biases  = (1 - self.tau) * target_layer_v.biases  + self.tau * layer_v.biases
        
        # advantage layer
        target_layer_a = self.target_model.advantage
        layer_a = self.model.advantage
        target_layer_a.weights = (1 - self.tau) * target_layer_a.weights + self.tau * layer_a.weights
        target_layer_a.biases  = (1 - self.tau) * target_layer_a.biases  + self.tau * layer_a.biases

class Network:
    
    def __init__(self, layerstruct, num_actions):
        
        # init all layers
        self.layers = []
        for i in range(len(layerstruct) - 1):
            self.layers.append(Layer(layerstruct[i], layerstruct[i + 1]))
        
        self.value = Layer(layerstruct[-1], 1) # value layer
        self.advantage = Layer(layerstruct[-1], num_actions) # advantage head
    
    def calculate(self, inputs):
        
        out = inputs
        
        # hidden layers
        for layer in self.layers: out = layer.calculate(out)
        
        # dueling function
        V = self.value.calculate(out) # shape(batch, 1)
        A = self.advantage.calculate(out) # shape(batch, num_actions)
        A_mean = np.mean(A, axis=1, keepdims=True)
        Q = V + (A - A_mean)
        
        # store values for bp
        self.last_Q = Q
        self.last_V = V
        self.last_A = A
        self.last_A_mean = A_mean
        self.last_output_of_trunk = out
        
        return Q
    
    def backprop(self, states, target, lr=0.0001):
        
        # forward pass
        Q_pred = self.calculate(states)
        
        # find loss
        loss = np.mean((Q_pred - target) ** 2)
        
        # math taken from online
        dL_dQ = (2 * (Q_pred - target)) / Q_pred.shape[0]
        dQ_dV = np.ones_like(self.last_V)  # shape (batch, 1)
        num_actions = self.last_A.shape[1]
        dQ_dA = np.ones_like(self.last_A) - (1 / num_actions)
        dL_dV = dL_dQ * dQ_dV     # shape (batch, 1)
        dL_dA = dL_dQ * dQ_dA     # shape (batch, num_actions)
        
        # execute bp
        dA_prev = self.advantage.backward(dL_dA)
        dV_prev = self.value.backward(dL_dV)
        
        # combine gradients
        dTrunk = dA_prev + dV_prev
        
        # backprop hidden layers
        for layer in reversed(self.layers): dTrunk = layer.backward(dTrunk)
        
        # update values
        for layer in self.layers: layer.update_parameters(lr)
        self.value.update_parameters(lr)
        self.advantage.update_parameters(lr)
        
        return loss
    
    def mse(pred, target): return np.mean((pred - target)**2) #loss function
    def d_mse(pred, target): return 2 * (pred - target) / pred.size #derivative loss function

class Layer:
    
    def __init__(self, numinputs, numneurons):
        
        # init starting values (currently random)
        self.weights = 0.1 * np.random.randn(numinputs, numneurons)
        self.biases = np.zeros([1, numneurons])
        
        # currently using relu, will look into gelu or tanh
        self.function = relu
        self.d_function = d_relu
        
        # store previous inputs for BP
        self.last_inputs = None
        self.last_base = None
        self.last_output = None
        
        # gradient functions
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.biases)
        
        # learning rate
        self.lr = 1
    
    def calculate(self, inputs):
        
        # forward pass
        self.last_inputs = inputs # input values
        self.last_base = np.dot(inputs, self.weights) + self.biases # initial values
        self.last_output = self.function(self.last_base) # activated values
        
        # saves values for backprop
        
        return self.last_output
    
    def bp(self): # switch to adam later
        self.weights -= self.lr * self.dW
        self.biases -= self.lr * self.db
    
    def backward(self, dA):
        
        # loss gradient
        dZ = dA * self.d_function(self.last_z)
        
        self.dW = self.last_inputs.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.weights.T
        
        return dA_prev
    
    def update_parameters(self, lr):
        self.weights -= lr * self.dW
        self.biases  -= lr * self.db

class ReplayBuffer:
    
    def __init__(self, capacity=50000):
        
        # deque automatically deletes the oldest entry once it runs out of space
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        
        # collects state
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self): # the internet told me to add len in this way
        return len(self.buffer)
    
    def sample(self, batch_size):
        
        # provides rl with sample of "experiences"
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

class Environment:
    
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size

def main():
    
    state_size = 0
    action_size = 0
    
    env = Environment(state_size, action_size)
    agent = DuelingDQN(state_size, action_size)
    
    episodes = 100
    max_steps = 100
    
    for i in range(episodes):
        
        state = env.reset()
        reward = 0
        
        for j in range(max_steps):
            
            agent.choose_action(state)
            
            

if __name__ == "__main__":
    main()