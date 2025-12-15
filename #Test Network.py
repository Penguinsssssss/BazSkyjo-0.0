#NN test page

import random
import math
import numpy as np #type: ignore
#seed = random.randint(0, 9999)
#np.random.seed(seed)
#print(f"Seed: {seed}")
np.random.seed(2)
import copy
from collections import deque

e = math.e

# activation functions
def sig(input): return 1 / (1 + np.exp(-input)) # sigmoid function
def relu(input): return np.maximum(0, input) # rectified linear
def expo(input): return np.exp(input) # exponent

# derivative activaiton functions
def d_relu(x): return (x > 0).astype(float) # rectified linear

class DuelingDQN:
    
    def __init__(self, state_size, action_size, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, tau=0.01):
        
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
        self.model = Network([state_size, 32], action_size)
        
        # target network
        self.target_model = copy.deepcopy(self.model)
    
    def choose_action(self, state): # epsilon greedy function
        
        # "exploration", allows the model to sometimes choose a fully random action to get itself out of local minima
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # "explotation", the model chooses the action it thinks is best and gets the results
        q_values = self.model.calculate(state)
        q_values = q_values.flatten()
        return int(np.argmax(q_values))
    
    def compute_targets(self, batch):
        
        # get batch
        states, actions, rewards, next_states, dones = batch
        
        q_vals = self.model.calculate(states)
        
        # main network chooses next actions
        main_next_q = self.model.calculate(next_states)
        best_actions = np.argmax(main_next_q, axis=1)
        
        # target network evaluates action
        target_next_q = self.target_model.calculate(next_states)
        targets = q_vals.copy()
        
        max_next_q = np.max(target_next_q, axis=1)
        
        for i in range(len(states)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + \
                    self.gamma * target_next_q[i, best_actions[i]]
                    
        return targets
    
    def train_step(self, batch_size=32):
        
        if len(self.replayBuffer) < batch_size:
            return # prevent NN from trying to take more samples then are available
        
        # retrieve a batch from the replaybuffer
        batch = self.replayBuffer.sample(batch_size)
        
        # unpack batch
        states, actions, rewards, next_states, dones = batch
        
        # generate training targets using bellman equation
        targets = self.compute_targets(batch)
        
        # run backwards on network
        self.model.backprop(states, targets, lr=self.lr)
        
        # reduce epsilon (exploration value)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # use soft updates to avoid instability
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
        
        # init v and a for Dueling DQN
        self.value = Layer(layerstruct[-1], 1) # value layer, shape (batch, 1)
        self.advantage = Layer(layerstruct[-1], num_actions) # advantage head shape (batch, num_actions)
    
    def calculate(self, inputs):
        
        out = inputs
        
        # hidden layers
        for layer in self.layers: out = layer.calculate(out)
        
        # dueling function
        V = self.value.calculate(out) # shape (batch, 1)
        A = self.advantage.calculate(out) # shape (batch, num_actions)
        A_mean = np.mean(A, axis=1, keepdims=True) # shape (batch, 1)
        Q = V + (A - A_mean) # shape (batch, num_actions)
        Q = np.squeeze(Q, axis=1) if Q.shape[1] == 1 else Q
        
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
        
        dQ = (2 * (Q_pred - target)) / Q_pred.shape[0]
        dV = np.sum(dQ, axis=1, keepdims=True)
        dA = dQ - np.mean(dQ, axis=1, keepdims=True)
        dA_prev = self.advantage.backward(dA)
        dV_prev = self.value.backward(dV)
        dTrunk = dA_prev + dV_prev
        for layer in reversed(self.layers):
            dTrunk = layer.backward(dTrunk)
        
        """
        dL_dQ = (2 * (Q_pred - target)) / Q_pred.shape[0]
        dQ_dV = np.ones_like(self.last_V) # intended shape (batch, 1)
        num_actions = self.last_A.shape[1]
        dQ_dA = np.ones_like(self.last_A) - (1 / num_actions) # intended shape (batch, num_outputs)
        
        dL_dV = np.sum(dL_dQ, axis=1, keepdims=True) # intended shape (batch, 1)
        dL_dA = dL_dQ * dQ_dA # intended shape (batch, num_actions)
        
        # execute bp (causes crash)
        dA_prev = self.advantage.backward(dL_dA)
        dV_prev = self.value.backward(dL_dV)
        
        # combine gradients
        dTrunk = dA_prev + dV_prev
        
        # backprop hidden layers
        for layer in reversed(self.layers): dTrunk = layer.backward(dTrunk)
        """
        
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
        self.weights = 0.1 * np.random.randn(numinputs, numneurons) # shape (inputs, neurons)
        self.biases = np.zeros([1, numneurons]) # shape (i, neurons)
        
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
    
    def calculate(self, inputs):
        
        # forward pass
        self.last_inputs = inputs # store input values
        self.last_base = np.dot(inputs, self.weights) + self.biases # store initial values
        self.last_output = self.function(self.last_base) # store activated values
        
        return self.last_output
    
    def backward(self, dA):
        
        # loss gradient
        dZ = dA * self.d_function(self.last_base) # matrix of gradients
        
        self.dW = self.last_inputs.T.dot(dZ) # gradient of weights, shape (inputs, num_neurons)
        self.db = np.sum(dZ, axis=0, keepdims=True) # gradient of bias, shape (inputs, 1)
        
        dA_prev = dZ.dot(self.weights.T) # store and return gradient to prev layer
        
        return dA_prev
    
    def update_parameters(self, lr):
        
        self.weights -= lr * self.dW
        self.biases  -= lr * self.db
        
        np.clip(self.dW, -1, 1, out=self.dW)
        np.clip(self.db, -1, 1, out=self.db)

class ReplayBuffer:
    
    def __init__(self, capacity=50000):
        
        # deque automatically deletes the oldest entry once it runs out of space
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        
        # collects state
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        
        # provides rl with sample of "experiences"
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones)) # shape (batch, state_size)

class Environment:
    
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
    
    def reset(self): pass
    def step(self, action): pass
    def get_reward(self): pass

class RPS_Env(Environment):
    
    def __init__(self):
        super().__init__(state_size=3, action_size=3)
        self.opponent_move = None  # store last move
    
    def reset(self):
        # opponent picks first
        self.opponent_move = random.randint(0, 2)
        return self.encode_state(self.opponent_move)
    
    def step(self, action):
        # reward according to RPS rules
        opp = self.opponent_move
        reward = self.get_reward(action, opp)
        
        # generate next opponent move (new state)
        next_opp = random.randint(0, 2)
        next_state = self.encode_state(next_opp)
        
        done = False  # Each RPS turn is independent, so never-ending episode
        self.opponent_move = next_opp
        
        return next_state, reward, done
    
    def encode_state(self, move):
        # convert 0/1/2 into one-hot vector
        one_hot = np.zeros(3)
        one_hot[move] = 1
        return one_hot
    
    def get_reward(self, agent, opp):
        # Standard RPS result logic (rock paper scissors - 0 1 2)
        if agent == opp:
            return 0      # tie
        if (agent == 0 and opp == 2) or \
            (agent == 1 and opp == 0) or \
            (agent == 2 and opp == 1):
            return 1      # win
        return -1         # lose
    
    def devGetOutput(self, action=random.randint(0, 2)):
        return self.encode_state(action)

class Coin_Env(Environment):
    
    def __init__(self):
        super().__init__(state_size=2, action_size=2)
        self.flip = None  # store coinflip
    
    def reset(self):
        self.flip = random.randint(0, 1)
        return self.encode_state(self.flip)
    
    def encode_state(self, move):
        # convert 0/1 into one-hot vector
        one_hot = np.zeros(2)
        one_hot[move] = 1
        return one_hot
    
    def step(self, action):
        # reward according to RPS rules
        coin = self.flip
        reward = self.get_reward(action, coin)
        
        # generate next opponent move (new state)
        next_coin = random.randint(0, 1)
        next_state = self.encode_state(next_coin)
        
        done = False  # Each RPS turn is independent, so never-ending episode
        self.opponent_move = next_coin
        
        return next_state, reward, done
    
    def get_reward(self, agent, opp):
        if agent == opp: # correct coin toss
            return 1
        return -1 # incorrect coin toss

class Dice_Env(Environment):
    def __init__(self, dicesize):
        super().__init__(state_size=dicesize, action_size=dicesize)
        self.flip = None  # store coinflip
        self.dicesize = dicesize
    
    def reset(self):
        self.flip = random.randint(0, self.dicesize - 1)
        return self.encode_state(self.flip)
    
    def encode_state(self, move):
        # convert 0/1 into one-hot vector
        one_hot = np.zeros(self.dicesize)
        one_hot[move] = 1
        return one_hot
    
    def step(self, action):
        # reward according to RPS rules
        coin = self.flip
        reward = self.get_reward(action, coin)
        
        # generate next opponent move (new state)
        next_coin = random.randint(0, self.dicesize - 1)
        next_state = self.encode_state(next_coin)
        
        done = False  # Each RPS turn is independent, so never-ending episode
        self.opponent_move = next_coin
        
        return next_state, reward, done
    
    def get_reward(self, guess, dice):
        if guess == dice: # correct coin toss
            return 1
        return -1 # incorrect coin toss

def main():
    
    env = Dice_Env(10)
    agent = DuelingDQN(env.state_size, env.action_size, epsilon_decay=0.99995, epsilon_min=0.01) #set agent size to fit env
    
    # one_hot does not need more then 1 step
    episodes = 100000
    max_steps = 1
    
    # visual indicator for impacient humans
    total_reward = 0
    num_logs = 50
    
    for i in range(episodes):
        
        state = env.reset()
        
        for _ in range(max_steps):
            
            # agent makes a descision
            action = agent.choose_action(state)
            
            # get results from action
            next_state, reward, done = env.step(action)
            
            state = state.flatten()
            next_state = next_state.flatten()
            reward = float(reward)
            reward = np.clip(reward, -1, 1)
            
            total_reward += reward
            
            # save state to replaybuffer
            agent.replayBuffer.push(state, action, reward, next_state, done)
            
            # run a training step
            agent.train_step(batch_size=32)
            
            # set to the next point in the game
            state = next_state
            
            # if game is over stop loop and start new game
            if done:
                break
        
        if i % (episodes / num_logs) == 0:
            
            print(f"Episode {i}: epsilon={agent.epsilon:.3f}, avg_reward={total_reward / (episodes / num_logs):.3f}")
            
            total_reward = 0
            
            # show example trial
            #test_state = env.encode_state(random.randint(0, env.state_size - 1))
            #print("Inputs", test_state, "Predicted Q:", agent.model.calculate(test_state))
    
    for i in range(env.state_size): # test all states (for one_hot envs)
        test_state = env.encode_state(i)
        solution = agent.model.calculate(test_state)
        guess = list(solution[0]).index(max(solution[0])) + 1
        success = list(solution[0]).index(max(solution[0])) == i
        print(f"State {i + 1}:", guess, success)
        print(solution)

if __name__ == "__main__":
    main()