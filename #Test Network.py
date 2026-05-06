#NN test page

import random
import math
import numpy as np #type: ignore
seed = random.randint(0, 9999)
np.random.seed(seed)
print(f"Seed: {seed}")
#np.random.seed(3)
import copy
from collections import deque
from pandas import DataFrame

e = math.e

# activation functions
def sig(input): return 1 / (1 + np.exp(-input)) # sigmoid function
def relu(input): return np.maximum(0, input) # rectified linear
def expo(input): return np.exp(input) # exponent

# derivative activaiton functions
def d_relu(x): return (x > 0).astype(float) # rectified linear

isDebug = True
AGENT_PATH = "c:\\users\\benjaminsullivan\\downloads\\checkpoint big brain.npz"
isLoading = isDebug
SAVE_PATH = AGENT_PATH
isSaving = not isDebug

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
        self.model = Network([state_size, 512, 512, 512, 64], action_size) # 256, 256, 32
        
        # target network
        self.target_model = copy.deepcopy(self.model)
    
    def choose_action(self, state, legal_actions, return_q=False): # epsilon greedy function
        
        q_values = self.model.calculate(state).flatten()
        
        # mask illegal actions by setting them to very negative
        masked_q = np.full_like(q_values, -1e9)
        masked_q[legal_actions] = q_values[legal_actions]
        
        # "exploration", allows the model to sometimes choose a fully random action to get itself out of local minima
        if (np.random.rand() < self.epsilon) and not isDebug:
            action = int(random.choice(legal_actions))
        
        # "explotation", the model chooses the action it thinks is best and gets the results
        else:
            # Exploitation: choose BEST LEGAL action
            action = int(np.argmax(masked_q))
        
        if return_q:
            return action, masked_q  # <- array of evals for all moves
        return action
    
    def compute_targets(self, batch):
        
        # get batch
        states, actions, rewards, next_states, dones, next_legal_actions = batch
        
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
                
                masked_next_q = np.full(self.action_size, -1e9)
                masked_next_q[next_legal_actions[i]] = main_next_q[i, next_legal_actions[i]]
                best_action = np.argmax(masked_next_q)
                targets[i, actions[i]] = rewards[i] + self.gamma * target_next_q[i, best_action]
                
        return targets
    
    def train_step(self, batch_size=32):
        
        if len(self.replayBuffer) < batch_size:
            return # prevent NN from trying to take more samples then are available
        
        # retrieve a batch from the replaybuffer
        batch = self.replayBuffer.sample(batch_size)
        
        # unpack batch
        states, actions, rewards, next_states, dones, next_legal_actions = batch
        
        # generate training targets using bellman equation
        targets = self.compute_targets(batch)
        
        # run backwards on network
        self.model.backprop(states, targets, lr=self.lr)
        
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
    
    def save(self, path):
        
        data = {}
        
        # main trunk
        for i, layer in enumerate(self.model.layers):
            data[f"m_layers_{i}_W"] = layer.weights
            data[f"m_layers_{i}_b"] = layer.biases
            
        # main heads
        data["m_value_W"] = self.model.value.weights
        data["m_value_b"] = self.model.value.biases
        data["m_adv_W"] = self.model.advantage.weights
        data["m_adv_b"] = self.model.advantage.biases
        
        # target trunk
        for i, layer in enumerate(self.target_model.layers):
            data[f"t_layers_{i}_W"] = layer.weights
            data[f"t_layers_{i}_b"] = layer.biases
            
        # target head
        data["t_value_W"] = self.target_model.value.weights
        data["t_value_b"] = self.target_model.value.biases
        data["t_adv_W"] = self.target_model.advantage.weights
        data["t_adv_b"] = self.target_model.advantage.biases
        
        # meta
        data["epsilon"] = np.array([self.epsilon], dtype=np.float32)
        data["epsilon_decay"] = np.array([self.epsilon_decay], dtype=np.float32)
        data["epsilon_min"] = np.array([self.epsilon_min], dtype=np.float32)
        data["gamma"] = np.array([self.gamma], dtype=np.float32)
        data["lr"] = np.array([self.lr], dtype=np.float32)
        data["tau"] = np.array([self.tau], dtype=np.float32)
        data["state_size"] = np.array([self.state_size], dtype=np.int32)
        data["action_size"] = np.array([self.action_size], dtype=np.int32)
        data["num_layers"] = np.array([len(self.model.layers)], dtype=np.int32)
        
        np.savez_compressed(path, **data)
    
    def load(self, path, load_target=True, load_epsilon=True):
        
        ckpt = np.load(path, allow_pickle=False)
        
        # sanity checks
        if int(ckpt["state_size"][0]) != self.state_size:
            raise ValueError(f"Checkpoint state_size {int(ckpt['state_size'][0])} != agent state_size {self.state_size}")
        if int(ckpt["action_size"][0]) != self.action_size:
            raise ValueError(f"Checkpoint action_size {int(ckpt['action_size'][0])} != agent action_size {self.action_size}")
        
        expected_layers = int(ckpt["num_layers"][0])
        if expected_layers != len(self.model.layers):
            raise ValueError(f"Checkpoint trunk layers {expected_layers} != agent trunk layers {len(self.model.layers)}")
        
        # main trunk
        for i, layer in enumerate(self.model.layers):
            layer.weights = ckpt[f"m_layers_{i}_W"]
            layer.biases  = ckpt[f"m_layers_{i}_b"]
            
        # main heads
        self.model.value.weights = ckpt["m_value_W"]
        self.model.value.biases  = ckpt["m_value_b"]
        self.model.advantage.weights = ckpt["m_adv_W"]
        self.model.advantage.biases  = ckpt["m_adv_b"]
        
        if load_target:
            # target trunk
            for i, layer in enumerate(self.target_model.layers):
                layer.weights = ckpt[f"t_layers_{i}_W"]
                layer.biases  = ckpt[f"t_layers_{i}_b"]
                
            # target heads
            self.target_model.value.weights = ckpt["t_value_W"]
            self.target_model.value.biases  = ckpt["t_value_b"]
            self.target_model.advantage.weights = ckpt["t_adv_W"]
            self.target_model.advantage.biases  = ckpt["t_adv_b"]
            
        if load_epsilon:
            self.epsilon = float(ckpt["epsilon"][0])

class Network:
    
    def __init__(self, layerstruct, num_actions):
        
        # init all layers
        self.layers = []
        for i in range(len(layerstruct) - 1):
            self.layers.append(Layer(layerstruct[i], layerstruct[i + 1], activation="relu"))
        
        # init v and a for Dueling DQN
        self.value = Layer(layerstruct[-1], 1, activation="linear") # value layer, shape (batch, 1)
        self.advantage = Layer(layerstruct[-1], num_actions, activation="linear") # advantage head shape (batch, num_actions)
    
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
    
    def mse(pred, target): return np.mean((pred - target)**2) # loss function
    def d_mse(pred, target): return 2 * (pred - target) / pred.size # derivative loss function

class Layer:
    
    def __init__(self, numinputs, numneurons, activation):
        
        # init starting values (currently random)
        self.weights = 0.1 * np.random.randn(numinputs, numneurons) # shape (inputs, neurons)
        self.biases = np.zeros([1, numneurons]) # shape (i, neurons)
        
        if activation == "relu":
            self.function = relu
            self.d_function = d_relu
        elif activation == "linear":
            self.function = lambda x: x
            self.d_function = lambda x: np.ones_like(x)
        
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
        
        np.clip(self.dW, -1, 1, out=self.dW)
        np.clip(self.db, -1, 1, out=self.db)
        
        self.weights -= lr * self.dW
        self.biases  -= lr * self.db

class ReplayBuffer:
    
    def __init__(self, capacity=50000):
        
        # deque automatically deletes the oldest entry once it runs out of space
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_legal_action):
        
        # collects state
        self.buffer.append((state, action, reward, next_state, done, next_legal_action))
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        
        # provides rl with sample of "experiences"
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_legal_actions = zip(*batch)
        
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones), # shape (batch, state_size)
                list(next_legal_actions))

class Environment:
    
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
    
    def reset(self): pass
    def step(self, action): pass
    def get_reward(self): pass

class Quixx_Env(Environment):
    
    def __init__(self, debug=False):
        
        super().__init__(state_size=51, action_size=13)
        
        self.sheet = None
        self.dice = None
        self.done = False
        self.legal_marks = None
        
        self.debug = debug
        
        self.reset()
    
    def dbg(self, msg):
        if self.debug: print(msg)
    
    def reset(self):
        
        # roll all 6 dice
        self.dice = [random.randint(1,6) for _ in range(6)] #white1, white2, red, yellow, green, blue
        self.dbg(f"Dice rolled: {self.dice}")
        
        # create blank sheet
        self.sheet = {
            "red": [0,0,0,0,0,0,0,0,0,0,0],
            "yellow": [0,0,0,0,0,0,0,0,0,0,0],
            "green": [0,0,0,0,0,0,0,0,0,0,0],
            "blue": [0,0,0,0,0,0,0,0,0,0,0],
            "penalties": 0
            }
        
        self.done = False
        
        return self.encode_state()
    
    def encode_state(self):
        
        # flatten current sheet info
        sheet_vals = (
            self.sheet["red"]
            + self.sheet["yellow"]
            + self.sheet["green"]
            + self.sheet["blue"]
            + [self.sheet["penalties"]]
        )
        
        # flatten all info to state
        state = sheet_vals + self.dice
        
        return np.array(state, dtype=float)
    
    def step(self, action):
        
        # return gamestate if done
        if self.done: return self.encode_state(), 0, True # done = true
        
        # init reward
        reward = 0
        
        if action == 12:
            
            self.sheet["penalties"] += 1
            reward = -0.5
            self.dbg("Action was SKIP/PENALTY")
            
        else:
            white = self.dice[0:2]
            color_dice = self.dice[2:]  # red, yellow, green, blue dice
            
            if action < 4:
                color = ["red","yellow","green","blue"][action]
                idx = sum(white)
            elif action == 4:
                color = "red"; idx = white[0] + color_dice[0]
            elif action == 5:
                color = "red"; idx = white[1] + color_dice[0]
            elif action == 6:
                color = "yellow"; idx = white[0] + color_dice[1]
            elif action == 7:
                color = "yellow"; idx = white[1] + color_dice[1]
            elif action == 8:
                color = "green"; idx = white[0] + color_dice[2]
            elif action == 9:
                color = "green"; idx = white[1] + color_dice[2]
            elif action == 10:
                color = "blue"; idx = white[0] + color_dice[3]
            elif action == 11:
                color = "blue"; idx = white[1] + color_dice[3]
            
            self.dbg(f"Trying to mark {color}, at number {idx}")
            
            if action in self.legal_marks:
                if color in ["red", "yellow"]:
                    # rows go left to right
                    if 1 in self.sheet[color]:
                        last_mark = max(i for i, v in enumerate(self.sheet[color]) if v == 1)
                        skips = (idx-2) - last_mark - 1
                    else:
                        skips = (idx-2)  # skipped everything before first mark
                        
                else:
                    # green / blue go right to left
                    if 1 in self.sheet[color]:
                        last_mark = min(i for i, v in enumerate(self.sheet[color]) if v == 1)
                        skips = last_mark - (idx-2) - 1
                    else:
                        skips = (len(self.sheet[color]) - 1) - (idx-2)  # skipped everything after first mark
                
                self.sheet[color][idx-2] = 1  # idx starts at 1
                reward = 0.3 - (0.2 * skips)
                
            else:
                self.sheet["penalties"] += 1
                reward = -0.5
                self.dbg("Illegal move, penalty applied")
        
        self.dbg(f"{self.sheet["red"]}")
        self.dbg(f"{self.sheet["yellow"]}")
        self.dbg(f"{self.sheet["green"]}")
        self.dbg(f"{self.sheet["blue"]}")
        self.dbg(f"Pens: {self.sheet["penalties"]}")
        
        if self.sheet["penalties"] >= 4: self.done = True
        
        # roll new dice for next turn
        self.dice = [random.randint(1,6) for _ in range(6)]
        self.dbg(f"Dice rolled: {self.dice}")
        
        return self.encode_state(), reward, self.done
    
    def get_legal_actions(self):
        
        # init legal moves (will be returned at end of function)
        legal = []
        
        def can_mark(color, idx):
            if color in ["red", "yellow"]:
                if idx != 10:
                    return sum(self.sheet[color][idx-2:]) == 0 # only allowed to mark rightmost squares
                else: return sum(self.sheet[color]) >= 5 # must have 5 marks to claim lock
            else:
                if idx != 10:
                    return sum(self.sheet[color][:idx-1]) == 0 # only allowed to mark leftmost squares
                else: return sum(self.sheet[color]) >= 5 # still must have 5 marks to claim lock
        
        # white+white
        ww_sum = self.dice[0] + self.dice[1] - 1  # convert to index
        for action_idx, color in enumerate(["red", "yellow", "green", "blue"]):
            if can_mark(color, ww_sum):
                legal.append(action_idx)
        
        # red
        if can_mark("red", self.dice[0] + self.dice[2] - 2): legal.append(4)
        if can_mark("red", self.dice[1] + self.dice[2] - 2): legal.append(5)
        
        # yellow
        if can_mark("yellow", self.dice[0] + self.dice[3] - 2): legal.append(6)
        if can_mark("yellow", self.dice[1] + self.dice[3] - 2): legal.append(7)
        
        # green
        if can_mark("green", self.dice[0] + self.dice[4] - 2): legal.append(8)
        if can_mark("green", self.dice[1] + self.dice[4] - 2): legal.append(9)
        
        # blue
        if can_mark("blue", self.dice[0] + self.dice[5] - 2): legal.append(10)
        if can_mark("blue", self.dice[1] + self.dice[5] - 2): legal.append(11)
        
        # penalty action is always legal
        legal.append(12)
        
        self.dbg(f"Legal marks: {legal}")
        
        return legal
    
    def score_game(self):
        
        # penalty is -5
        score = self.sheet["penalties"] * -5
        
        # each row is worth more for every new mark
        point_vals = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78]
        
        for color in ["red", "yellow", "green", "blue"]:
            score += point_vals[sum(self.sheet[color])]
        
        self.dbg(f"Game over! Score is: {score}")
        
        return score

class Skyjo_Env(Environment):
    
    def __init__(self, debug=False):
        
        super().__init__(state_size=30, action_size=24)
        # 12 cards, 1 for discard, 1 for number of players, 1 for avg value of deck, 1 for lowest unknowns of any player, 1 for pending card, 1 for phase
        
        self.debug = debug
        
        self.reset()
    
    def reset(self):
        
        # make deck and discard
        self.deck = []
        for i in range(5): self.deck.append(-2) # five -2's
        for i in range(10): self.deck.append(-1) # ten -1's
        for i in range(15): self.deck.append(0) # fifteen 0's
        for i in range(12): # ten of 1 -> 12
            for j in range(10): self.deck.append(i + 1)
        random.shuffle(self.deck)
        self.discard = self.deck.pop()
        
        # create cards for other simulated players
        self.numplayers = random.randint(2, 6)
        self.hands = []
        for i in range(self.numplayers - 1):
            self.hands.append([self.deck.pop(), self.deck.pop(), None, None, None, None, None, None, None, None, None, None])
        
        # cards for self
        self.hand = [self.deck.pop(), self.deck.pop(), None, None, None, None, None, None, None, None, None, None]
        self.phand = copy.deepcopy(self.hand)
        
        self.phase = "main"
        self.pendingcard = None
        
        self.done = False
        
        self.bonus = 0
        
        return self.encode_state()
    
    def norm(self, card):
        if card is None: return 1.0 # sentinel
        else: return (card + 2) / 15 # norm value
    
    def encode_state(self):
        
        own = [self.norm(i) for i in self.hand]
        avg = self.norm(sum(self.deck) / len(self.deck))
        
        doubles = []
        for i in range(4):
        
            if ((self.hand[i] == self.hand[i + 4]) or (self.hand[i+8] == self.hand[i + 4]) or (self.hand[i] == self.hand[i + 8])) and (self.hand[i] is not None) and (self.hand[i+4] is not None): # if column is same card and not unknowns
                doubles.append(1)
            else: doubles.append(0)
        
        
        triples = []
        
        # check for triple
        for i in range(4):
        
            if self.hand[i] == self.hand[i + 4] == self.hand[i + 8] and self.hand[i] is not None: # if column is same card and not unknowns
                triples.append(1)
            else: triples.append(0)
        
        sums = []
        for i in range(4):
            sums.append((self.hand[i] if self.hand[i] is not None else avg + self.hand[i+4] if self.hand[i+4] is not None else avg + self.hand[i+8] if self.hand[i+8] is not None else avg + 6) / 45)
        
        return (
            own + # own cards
            [self.norm(self.discard)] + # dicard card
            [self.numplayers / 6] + # number of players
            [avg] + # average value of unknown
            [(min(sum(card is None for card in hand) for hand in self.hands)) / 12] + # lowest number of unknowns for any opposing player
            [self.norm(self.pendingcard) if self.phase == "pending" else 1.0] + # gives bot pending card if held
            [1 if self.phase == "pending" else 0] +
            doubles +
            triples +
            sums
        )
    
    def step(self, action):
        
        if self.phase == "main":
            self.dbg("")
            self.dbg(f"baz hand: {self.hand[:4]}")
            self.dbg(f"baz hand: {self.hand[4:8]}")
            self.dbg(f"baz hand: {self.hand[8:]}")
            self.dbg(f"discard: {self.discard}")
        
        # if any players have 0 unknowns or deck is empty
        if min(sum(card is None for card in hand) for hand in self.hands) == 0 or sum(card is None for card in self.hand) == 0 or len(self.deck) == 0:
            
            # end game
            self.done = True
            
            # return state
            return self.encode_state(), 0, self.done
        
        # init reward
        self.reward = 0
        
        prior_phase = self.phase
        self.phand = copy.deepcopy(self.hand)
        
        # agent acts
        self.act(action)
        
        # calculate reward
        if prior_phase == "main" and action == 12:
            self.reward = 0.0
        else:
            self.reward = self.calcreward()
        
        # return state if pending
        if self.phase == "pending":
            return self.encode_state(), self.reward, False
        
        else: # advance opponents if not pending
            self.advanceopp()
        
        return self.encode_state(), self.reward, self.done
    
    def act(self, action):
        
        # main: 0-11 discard -> hand, 12 reveal top of deck
        # pending: 0-11 pending -> hand, 12-23 pending -> discard + reveal one card in hand
        if self.phase == "main":
            
            if action == 0: # TAKE FROM DRAW
                
                # draw card from deck and query dqn
                
                self.pendingcard = self.deck.pop()
                
                self.dbg(f"Action 0 (draw) chosen (card {self.pendingcard})")
            
            else: # TAKE FROM DISCARD
                
                self.dbg(f"Action 1 chosen (take discard) discard is: {self.discard}")
                #self.dbg("\n------------------------------")
                
                # store discard
                self.pendingcard = self.discard
            
            # set phase to pending after taking a card
            self.phase = "pending"
        
        elif self.phase == "pending":
            
            if action < 12: # ACCEPT CARD
                
                self.bonus = self.calcbonus(action, self.hand[action], self.pendingcard, self.hand[action] is None)
                
                self.dbg(f"chosen to accept pendingcard to slot {action}")
                self.dbg("\n------------------------------")
                
                # place the unknown into the discard
                if self.hand[action] is None:
                    self.discard = self.deck.pop()
                else: self.discard = self.hand[action] # place known card into discard
                
                self.hand[action] = self.pendingcard
            
            else: # REJECT CARD
                
                self.bonus = self.calcbonus(action - 12, None, 0, True)
                
                self.dbg(f"chosen to reject pendingcard and reveal slot {action - 12}")
                self.dbg("\n------------------------------")
                
                # discard rejected card
                self.discard = self.pendingcard
                
                # reveal card
                self.hand[action - 12] = self.deck.pop()
            
            self.pendingcard = None
            self.phase = "main"
            if isDebug: input()
    
    def advanceopp(self):
        
        for i in self.hands: # basic opponent bot
            
            if len(self.deck)  == 0: return # ensure deck has cards
            
            avg = sum(self.deck) / len(self.deck)
            
            if self.discard < avg: # take discard if is lower than average unknown
                
                # replace random unknown/card with higher value
                
                replaceindexes = []
                for j in range(len(i)): # for card in cards
                    if (i[j] if i[j] is not None else avg) >= self.discard: # if card is more than discard (treat unknown like avg)
                        replaceindexes.append(j) # add it to indexes
                
                replacedidx = random.choice(replaceindexes)
                replacedcard = i[replacedidx] if i[replacedidx] is not None else self.deck.pop()
                
                i[replacedidx] = self.discard
                self.discard = replacedcard
            
            else: # draw card
                
                # replace random unknown/card with higher value
                
                replaceindexes = []
                for j in range(len(i)): # for card in cards
                    if (i[j] if i[j] is not None else float("inf")) >= self.deck[-1]: # if card is more than discard (treat unkown like inf)
                        replaceindexes.append(j) # add it to indexes
                
                replacedidx = random.choice(replaceindexes)
                replacedcard = i[replacedidx] if i[replacedidx] is not None else self.deck.pop(0)
                
                i[replacedidx] = self.deck.pop()
                self.discard = replacedcard
    
    def calcbonus(self, slot, old, new, wasUnknown):
        
        bonus = 0
        
        if wasUnknown:
            bonus += 0
        else:
            bonus += ((old*abs(old)) - (new*abs(new))) / 60
        
        row = []
        for i in range(3):
            row.append(self.hand[(slot % 4) + (i * 4)])
        
        if row.count(new) >= 2:
            bonus += 1
            if row.count(new) == 3:
                bonus += 10
        
        return bonus
    
    def calcreward(self):
        
        avg = sum(self.deck) / len(self.deck)
        
        pExpected = 0
        for i in range(4):
        
            card1 = self.phand[i]
            card2 = self.phand[i + 4]
            card3 = self.phand[i + 8]
            
            if card1 == card2 == card3 and card1 is not None: # if column is same card and not unknowns
                pExpected += 0
            else: # otherwise
                pExpected += card1 if card1 is not None else avg # add either value of card or average of unknowns
                pExpected += card2 if card2 is not None else avg
                pExpected += card3 if card3 is not None else avg
        
        cExpected = 0
        for i in range(4):
        
            card1 = self.hand[i]
            card2 = self.hand[i + 4]
            card3 = self.hand[i + 8]
            
            if card1 == card2 == card3 and card1 is not None: # if column is same card and not unknowns
                cExpected += 0
            else: # otherwise
                cExpected += card1 if card1 is not None else avg # add either value of card or average of unknowns
                cExpected += card2 if card2 is not None else avg
                cExpected += card3 if card3 is not None else avg
        
        reward = pExpected - cExpected
        reward += self.bonus
        
        if self.done:
            self.reward += -self.score_game()
        
        return reward
    
    def get_legal_actions(self):
        
        if self.phase == "main":
            
            return list(range(2))  # 0-11 discard->hand, 12 draw
        
        else:
            
            legal = list(range(12))  # accept pending into slot 0-11 always legal
            
            # reject+reveal only legal if that reveal slot is None
            
            for i in range(12):
                if self.hand[i] is None:
                    legal.append(12 + i)
                
            #print(legal)
            return legal
    
    def score_game(self):
        
        score = 0
        unknown = sum(self.deck) / len(self.deck) if len(self.deck) > 0 else 5
        
        for i in range(4):
        
            if self.hand[i] == self.hand[i + 4] == self.hand[i + 8] and self.hand[i] is not None: # if column is same card and not unknowns
                pass # add zero to score
            else: # otherwise
                score += self.hand[i] if self.hand[i] is not None else unknown # add either value of card or average of unknowns
                score += self.hand[i + 4] if self.hand[i + 4] is not None else unknown
                score += self.hand[i + 8] if self.hand[i + 8] is not None else unknown
        
        if self.debug: self.dbg(f"Score: {score}")
        return score
    
    def dbg(self, msg):
        if self.debug: print(msg)

def main():
    
    print("program running")
    
    # env and model settings
    env = Skyjo_Env(debug=isDebug)
    agent = DuelingDQN(env.state_size, env.action_size, epsilon_decay=0.9995, epsilon_min=0.01) # set agent size to fit env
    if isLoading: agent.load(AGENT_PATH, load_epsilon=True) # else start fresh with a new agent
    
    episodes = 1 if isDebug else 10000
    max_steps = 1000
    
    # data collection settings
    episode_scores = []
    total_episode_score = 0
    episode_reward = 0
    rewards = []
    scores = []
    
    # visual indicator for impacient humans
    num_logs = 1000
    
    for ep in range(episodes):
        
        env.debug = (ep == episodes-1)
        state = np.array(env.reset(), dtype=np.float32)
        
        for _ in range(max_steps):
            
            # agent makes a descision
            legal_actions = env.get_legal_actions()
            
            action, values = agent.choose_action(state, legal_actions=legal_actions, return_q=True)
            
            # step env
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            reward = float(reward)
            
            episode_reward += reward
            
            next_legal_actions = env.get_legal_actions() if not done else []
            
            # save state to replaybuffer
            agent.replayBuffer.push(state, action, reward, next_state, done, next_legal_actions)
            
            # run a training step
            agent.train_step(batch_size=32)
            
            # set to the next point in the game
            state = next_state
            
            # if game is over stop loop and start new game
            if done:
                break
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        score = env.score_game()
        episode_scores.append(score)
        total_episode_score += score
        
        if ep % 10 == 0: # data tracking
            rewards.append(reward)
            scores.append(score)
        
        if ep % (episodes / num_logs) == 0 and ep != 0: # logs
            
            print(f"Episode {ep}: epsilon={agent.epsilon:.3f}, avg_reward={episode_reward / (episodes / num_logs):.3f}, avg_game_score={total_episode_score/len(episode_scores)}")
            
            episode_reward = 0
            
            episode_scores = []
            total_episode_score = 0
            
            if isSaving: agent.save(SAVE_PATH)

if __name__ == "__main__":
    main()

""" EXAMPLE OF WEIRD BEHAVIOR
Seed: 663
program running

baz hand: [7, 1, None, None]
baz hand: [None, None, None, None]    
baz hand: [None, None, None, None]    
discard: 5
Action 0 (draw) chosen (card 7)       
chosen to accept pendingcard to slot 0

------------------------------        


baz hand: [7, 1, None, None]
baz hand: [None, None, None, None]    
baz hand: [None, None, None, None]    
discard: 9
Action 0 (draw) chosen (card 12)      
chosen to accept pendingcard to slot 0

------------------------------        


baz hand: [12, 1, None, None]
baz hand: [None, None, None, None]    
baz hand: [None, None, None, None]    
discard: 10
Action 0 (draw) chosen (card 12)      
chosen to accept pendingcard to slot 0

------------------------------        


baz hand: [12, 1, None, None]     
baz hand: [None, None, None, None]
baz hand: [None, None, None, None]
discard: 6
Action 0 (draw) chosen (card 6)
chosen to accept pendingcard to slot 0

------------------------------


baz hand: [6, 1, None, None]
baz hand: [None, None, None, None]
baz hand: [None, None, None, None]
discard: 10
Action 0 (draw) chosen (card 4)
chosen to accept pendingcard to slot 0

------------------------------


baz hand: [4, 1, None, None]
baz hand: [None, None, None, None]
baz hand: [None, None, None, None]
discard: 1
Action 1 chosen (take discard) discard is: 1
chosen to accept pendingcard to slot 2

------------------------------


baz hand: [4, 1, 1, None]
baz hand: [None, None, None, None]
baz hand: [None, None, None, None]
discard: 2
Action 1 chosen (take discard) discard is: 2
chosen to accept pendingcard to slot 0

------------------------------


baz hand: [2, 1, 1, None]
baz hand: [None, None, None, None]
baz hand: [None, None, None, None]
discard: 6
Action 0 (draw) chosen (card 8)
chosen to reject pendingcard and reveal slot 11

------------------------------


baz hand: [2, 1, 1, None]
baz hand: [None, None, None, None]
baz hand: [None, None, None, 11]
discard: 9
Action 0 (draw) chosen (card 10)
chosen to accept pendingcard to slot 3

------------------------------


baz hand: [2, 1, 1, 10]
baz hand: [None, None, None, None]
baz hand: [None, None, None, 11]
discard: 1
Action 1 chosen (take discard) discard is: 1
chosen to accept pendingcard to slot 3      

------------------------------


baz hand: [2, 1, 1, 1]
baz hand: [None, None, None, None]    
baz hand: [None, None, None, 11]      
discard: 11
Action 0 (draw) chosen (card 0)       
chosen to accept pendingcard to slot 6

------------------------------        


baz hand: [2, 1, 1, 1]
baz hand: [None, None, 0, None]
baz hand: [None, None, None, 11]
discard: 12
Action 0 (draw) chosen (card 8)
chosen to reject pendingcard and reveal slot 9

------------------------------


baz hand: [2, 1, 1, 1]
baz hand: [None, None, 0, None]
baz hand: [None, -1, None, 11]
discard: 9
Action 0 (draw) chosen (card 0)
chosen to accept pendingcard to slot 4

------------------------------


baz hand: [2, 1, 1, 1]
baz hand: [0, None, 0, None]
baz hand: [None, -1, None, 11]
discard: 7
Action 0 (draw) chosen (card 7)
chosen to reject pendingcard and reveal slot 10

------------------------------


baz hand: [2, 1, 1, 1]
baz hand: [0, None, 0, None]
baz hand: [None, -1, 5, 11]
discard: 1
Action 1 chosen (take discard) discard is: 1
chosen to accept pendingcard to slot 5

------------------------------


baz hand: [2, 1, 1, 1]
baz hand: [0, 1, 0, None]
baz hand: [None, -1, 5, 11]
discard: 2
Score: 33.325581395348834
"""

"""
Episode 100: epsilon=0.951, avg_reward=22.917, avg_game_score=62.20736142397325
Episode 200: epsilon=0.904, avg_reward=5.949, avg_game_score=63.49193643209338
Episode 300: epsilon=0.860, avg_reward=41.209, avg_game_score=59.918043408188396
Episode 400: epsilon=0.818, avg_reward=36.518, avg_game_score=58.5174557875186
Episode 500: epsilon=0.778, avg_reward=35.183, avg_game_score=57.874610522368805
Episode 600: epsilon=0.740, avg_reward=30.912, avg_game_score=58.61300736444958
Episode 700: epsilon=0.704, avg_reward=48.467, avg_game_score=55.673812938202545
Episode 1600: epsilon=0.449, avg_reward=106.802, avg_game_score=49.287830645580524603                                                                       8355
Episode 1700: epsilon=0.427, avg_reward=101.043, avg_game_score=47.57738998896054388624                                                                      72
Episode 1800: epsilon=0.406, avg_reward=103.993, avg_game_score=48.917126987255353534                                                                      9282
Episode 1900: epsilon=0.386, avg_reward=109.479, avg_game_score=47.171802884813189444                                                                       3288
Episode 2000: epsilon=0.368, avg_reward=115.109, avg_game_score=46.63279900746763       
Episode 2100: epsilon=0.350, avg_reward=130.684, avg_game_score=44.4612071001415        
Episode 2200: epsilon=0.333, avg_reward=133.569, avg_game_score=45.28799937544024       
Episode 2300: epsilon=0.316, avg_reward=119.298, avg_game_score=46.06359462875291       
Episode 2400: epsilon=0.301, avg_reward=123.850, avg_game_score=45.39479670696697       
Episode 2500: epsilon=0.286, avg_reward=124.104, avg_game_score=45.50637402805091       
Episode 2600: epsilon=0.272, avg_reward=134.636, avg_game_score=43.86509009518008       
Episode 2700: epsilon=0.259, avg_reward=116.228, avg_game_score=46.40203502293707       
Episode 2800: epsilon=0.246, avg_reward=125.410, avg_game_score=45.6886547494982
Episode 2900: epsilon=0.234, avg_reward=139.801, avg_game_score=45.52583906869402
Episode 3000: epsilon=0.223, avg_reward=126.701, avg_game_score=45.868907662744185
Episode 3100: epsilon=0.212, avg_reward=148.628, avg_game_score=44.38179654430128
Episode 3200: epsilon=0.202, avg_reward=160.407, avg_game_score=43.581100156305325
Episode 3300: epsilon=0.192, avg_reward=150.847, avg_game_score=42.154951571824746
Episode 3400: epsilon=0.183, avg_reward=146.649, avg_game_score=43.9246487631625
Episode 3500: epsilon=0.174, avg_reward=154.558, avg_game_score=42.34712513661359
Episode 3600: epsilon=0.165, avg_reward=151.229, avg_game_score=45.541040029288006
Episode 3700: epsilon=0.157, avg_reward=162.648, avg_game_score=43.58571520615299
Episode 3800: epsilon=0.149, avg_reward=145.970, avg_game_score=41.95216344221138
Episode 3900: epsilon=0.142, avg_reward=160.241, avg_game_score=42.189632556338694
Episode 4000: epsilon=0.135, avg_reward=175.034, avg_game_score=40.726473335973054
Episode 4100: epsilon=0.129, avg_reward=188.014, avg_game_score=40.52867224402437
Episode 4200: epsilon=0.122, avg_reward=172.673, avg_game_score=41.1131126347178
Episode 4300: epsilon=0.116, avg_reward=166.961, avg_game_score=42.14628996756894
Episode 4400: epsilon=0.111, avg_reward=175.820, avg_game_score=40.22487373280322
Episode 4500: epsilon=0.105, avg_reward=169.503, avg_game_score=40.679736432316616
Episode 4600: epsilon=0.100, avg_reward=155.307, avg_game_score=41.4257863302418
Episode 4700: epsilon=0.095, avg_reward=170.475, avg_game_score=43.488228539222376
Episode 4800: epsilon=0.091, avg_reward=163.409, avg_game_score=44.1445303411669
Episode 4900: epsilon=0.086, avg_reward=160.063, avg_game_score=41.54665343147094
Episode 5000: epsilon=0.082, avg_reward=183.165, avg_game_score=41.68448517134585
Episode 5100: epsilon=0.078, avg_reward=184.073, avg_game_score=40.972212350595534
Episode 5200: epsilon=0.074, avg_reward=189.533, avg_game_score=42.15726094512238
Episode 5300: epsilon=0.071, avg_reward=151.270, avg_game_score=41.873834217216434
Episode 5400: epsilon=0.067, avg_reward=188.744, avg_game_score=39.33045617167505
Episode 5500: epsilon=0.064, avg_reward=210.797, avg_game_score=40.41452049603706
Episode 5600: epsilon=0.061, avg_reward=217.982, avg_game_score=37.25239355299875
Episode 5700: epsilon=0.058, avg_reward=189.316, avg_game_score=39.30069709459023
Episode 5800: epsilon=0.055, avg_reward=188.436, avg_game_score=38.14250070076673
Episode 5900: epsilon=0.052, avg_reward=214.966, avg_game_score=36.05841693325761
Episode 6000: epsilon=0.050, avg_reward=217.971, avg_game_score=36.516627744330656
Episode 6100: epsilon=0.047, avg_reward=219.392, avg_game_score=40.47136115475816
Episode 6200: epsilon=0.045, avg_reward=214.317, avg_game_score=39.77284513579341
Episode 6300: epsilon=0.043, avg_reward=213.587, avg_game_score=38.35737327214989
Episode 6400: epsilon=0.041, avg_reward=196.285, avg_game_score=37.956601932512605
Episode 6500: epsilon=0.039, avg_reward=218.846, avg_game_score=38.83430323505834
Episode 6600: epsilon=0.037, avg_reward=216.593, avg_game_score=36.21350507624813
Episode 6700: epsilon=0.035, avg_reward=212.060, avg_game_score=38.889846309958386
Episode 6800: epsilon=0.033, avg_reward=225.472, avg_game_score=39.74088176541907
Episode 6900: epsilon=0.032, avg_reward=185.046, avg_game_score=40.785062334765755
Episode 7000: epsilon=0.030, avg_reward=213.998, avg_game_score=40.40014713308185
Episode 7100: epsilon=0.029, avg_reward=170.210, avg_game_score=42.67988049834996
Episode 7200: epsilon=0.027, avg_reward=186.375, avg_game_score=41.19373883656558
Episode 7300: epsilon=0.026, avg_reward=214.835, avg_game_score=38.90763503724184
Episode 7400: epsilon=0.025, avg_reward=221.063, avg_game_score=40.579010278348115
Episode 7500: epsilon=0.023, avg_reward=236.495, avg_game_score=39.373934346169584
Episode 7600: epsilon=0.022, avg_reward=220.549, avg_game_score=37.983738117297456
Episode 7700: epsilon=0.021, avg_reward=250.958, avg_game_score=39.43423810947665
Episode 7800: epsilon=0.020, avg_reward=245.144, avg_game_score=39.193913400490494
Episode 7900: epsilon=0.019, avg_reward=231.213, avg_game_score=39.97099161878593
Episode 8000: epsilon=0.018, avg_reward=257.350, avg_game_score=40.5361296778014
Episode 8100: epsilon=0.017, avg_reward=226.561, avg_game_score=40.40612980644958
Episode 8200: epsilon=0.017, avg_reward=251.815, avg_game_score=39.674566706052225
Episode 8300: epsilon=0.016, avg_reward=235.647, avg_game_score=40.36630267041188
Episode 8400: epsilon=0.015, avg_reward=258.577, avg_game_score=39.794342714662235
Episode 8500: epsilon=0.014, avg_reward=253.086, avg_game_score=40.63886799753824
Episode 8600: epsilon=0.014, avg_reward=249.817, avg_game_score=41.491157938790494
Episode 8700: epsilon=0.013, avg_reward=241.094, avg_game_score=42.25454295101543
Episode 8800: epsilon=0.012, avg_reward=245.392, avg_game_score=40.95150197584523
Episode 8900: epsilon=0.012, avg_reward=217.671, avg_game_score=41.741734431083515
Episode 9000: epsilon=0.011, avg_reward=231.175, avg_game_score=42.687873688967564
Episode 9100: epsilon=0.011, avg_reward=264.251, avg_game_score=42.28086517408414
Episode 9200: epsilon=0.010, avg_reward=245.467, avg_game_score=42.604835524096
Episode 9300: epsilon=0.010, avg_reward=254.724, avg_game_score=40.40003082790415
Episode 9400: epsilon=0.010, avg_reward=262.379, avg_game_score=41.7371292679038
Episode 9500: epsilon=0.010, avg_reward=257.748, avg_game_score=41.02632595840634
Episode 9600: epsilon=0.010, avg_reward=257.879, avg_game_score=42.21699876414114
Episode 9700: epsilon=0.010, avg_reward=261.834, avg_game_score=40.92997108533909
Episode 9800: epsilon=0.010, avg_reward=251.532, avg_game_score=42.989320765464896
Episode 9900: epsilon=0.010, avg_reward=264.672, avg_game_score=42.63955809290399

"""