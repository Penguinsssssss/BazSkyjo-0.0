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
from pandas import DataFrame

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
        self.model = Network([state_size, 128, 128, 32], action_size)
        
        # target network
        self.target_model = copy.deepcopy(self.model)
    
    def choose_action(self, state, legal_actions, return_q=False): # epsilon greedy function
        
        q_values = self.model.calculate(state).flatten()
        
        # mask illegal actions by setting them to very negative
        masked_q = np.full_like(q_values, -1e9)
        masked_q[legal_actions] = q_values[legal_actions]
        
        # "exploration", allows the model to sometimes choose a fully random action to get itself out of local minima
        if np.random.rand() < self.epsilon:
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
        
        super().__init__(state_size=42, action_size=24)
        # 12 cards, 12 for upstream player, 12 for downstream player, 1 for discard, 1 for number of players, 1 for avg value of deck, 1 for lowest unknowns of any player, 1 for pending card, 1 for phase
        
        self.debug = debug
        
        self.reset()
    
    def reset(self):
        
        # create cards for other simulated players
        self.numplayers = random.randint(2, 6)
        self.hands = []
        for i in range(self.numplayers - 1):
            self.hands.append([None, None, None, None, None, None, None, None, None, None, None, None])
        
        # cards for self
        self.hand = [None, None, None, None, None, None, None, None, None, None, None, None]
        
        # make deck and discard
        self.deck = []
        for i in range(5): self.deck.append(-2) # five -2's
        for i in range(10): self.deck.append(-1) # ten -1's
        for i in range(15): self.deck.append(0) # fifteen 0's
        for i in range(12): # ten of 1 -> 12
            for j in range(10): self.deck.append(i + 1)
        random.shuffle(self.deck)
        self.discard = self.deck.pop()
        
        self.phase = "main"
        self.pendingcard = None
        
        self.done = False
        
        return self.encode_state()
    
    def encode_state(self):
        
        own = [self.enc_card(i) for i in self.hand]
        down = [self.enc_card(i) for i in self.hands[0]]
        up = [self.enc_card(i) for i in self.hands[-1]]
        
        return (
            own + # own cards
            down + # downstream player's cards
            up + # upstream player's cards
            [self.discard] + # dicard card
            [self.numplayers] + # number of players
            [sum(self.deck) / len(self.deck)] + # average value of unknown
            [min(sum(card is None for card in hand) for hand in self.hands)] + # lowest number of unknowns for any opposing player
            [self.pendingcard if self.phase == "pending" else 13] + # gives bot pending card if held
            [1 if self.phase == "pending" else 0]
        )
    
    def enc_card(self, card):
            return float(card) if card is not None else 13.0  # sentinel for unknown
    
    def step(self, action):
        
        self.dbg(f"baz hand: {self.hand[:4]}")
        self.dbg(f"baz hand: {self.hand[4:8]}")
        self.dbg(f"baz hand: {self.hand[8:]}")
        self.dbg("")
        
        # if any players have 0 unknowns or deck is empty
        if min(sum(card is None for card in hand) for hand in self.hands) == 0 or sum(card is None for card in self.hand) == 0 or len(self.deck) == 0:
            
            # end game
            self.done = True
            
            # return state
            return self.encode_state(), 0, self.done
        
        # init reward
        self.reward = 0
        
        # agent acts
        self.act(action)
        
        # calculate reward
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
            
            if action == 12: # TAKE FROM DRAW
                
                # draw card from deck and query dqn
                self.phase = "pending"
                self.pendingcard = self.deck.pop()
                
                self.dbg(f"Action 12 (draw) chosen (card {self.pendingcard})")
                
                return
            
            else: # TAKE FROM DISCARD
                
                self.dbg(f"Action {action} chosen (discard to x) discard is: {self.discard}")
                
                # store discard
                card = self.discard
                
                # place the unknown into the discard
                if self.hand[action] is None:
                    self.discard = self.deck.pop()
                else: self.discard = self.hand[action] # place known card into discard
                
                # change hand
                self.hand[action] = card
                
                self.phase = "main"
        
        elif self.phase == "pending":
            
            if action < 12: # ACCEPT CARD
                
                self.dbg(f"chosen to accept pendingcard to slot {action}")
                
                # place the unknown into the discard
                if self.hand[action] is None:
                    self.discard = self.deck.pop()
                else: self.discard = self.hand[action] # place known card into discard
                
                self.hand[action] = self.pendingcard
            
            else: # REJECT CARD
                
                self.dbg(f"chosen to reject pendingcard and reveal slot {action - 12}")
                
                # discard rejected card
                self.discard = self.pendingcard
                
                # reveal card
                self.hand[action - 12] = self.deck.pop()
            
            self.pendingcard = None
            self.phase = "main"
    
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
    
    def calcreward(self):
        
        reward = 0
        avg = sum(self.deck) / len(self.deck)
        
        for i in range(4):
        
            if self.hand[i] == self.hand[i + 4] == self.hand[i + 8] and self.hand[i] is not None: # if column is same card and not unknowns
                pass # add zero to score
            else: # otherwise
                reward += self.hand[i] if self.hand[i] is not None else avg + 5 # add either value of card or average of unknowns
                reward += self.hand[i + 4] if self.hand[i + 4] is not None else avg + 5
                reward += self.hand[i + 8] if self.hand[i + 8] is not None else avg + 5
        
        return -reward
    
    def get_legal_actions(self):
        
        if self.phase == "main":
            
            return list(range(13))  # 0-11 discard->hand, 12 draw
        
        else:
            
            legal = list(range(12))  # accept pending into slot 0-11 always legal
            
            # reject+reveal only legal if that reveal slot is None
            
            for i in range(12):
                if self.hand[i] is None:
                    legal.append(12 + i)
                
            return legal
    
    def score_game(self):
        
        score = 0
        
        for i in range(4):
        
            if self.hand[i] == self.hand[i + 4] == self.hand[i + 8] and self.hand[i] is not None: # if column is same card and not unknowns
                pass # add zero to score
            else: # otherwise
                score += self.hand[i] if self.hand[i] is not None else self.deck.pop() # add either value of card or average of unknowns
                score += self.hand[i + 4] if self.hand[i + 4] is not None else self.deck.pop()
                score += self.hand[i + 8] if self.hand[i + 8] is not None else self.deck.pop()
                
        return score
    
    def dbg(self, msg):
        if self.debug: print(msg)

def main():
    
    # env and model settings
    env = Skyjo_Env(debug=False)
    agent = DuelingDQN(env.state_size, env.action_size, epsilon_decay=0.9999995, epsilon_min=0.01) # set agent size to fit env
    #agent.load("c:\\users\\benjaminsullivan\\downloads\\checkpoint3.npz")
    
    episodes = 10000
    max_steps = 1000
    
    # data collection settings
    episode_scores = []
    total_episode_score = 0
    episode_reward = 0
    rewards = []
    scores = []
    
    # visual indicator for impacient humans
    num_logs = 100
    
    for ep in range(episodes):
        
        env.debug = (ep == episodes-1)
        state = env.reset()
        
        for _ in range(max_steps):
            
            # agent makes a descision
            legal_actions = env.get_legal_actions()
            action, values = agent.choose_action(state, legal_actions=legal_actions, return_q=True)
            
            # step env
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            reward = float(reward)
            
            episode_reward += reward
            
            # save state to replaybuffer
            agent.replayBuffer.push(state, action, reward, next_state, done)
            
            # run a training step
            agent.train_step(batch_size=32)
            
            # set to the next point in the game
            state = next_state
            
            # if game is over stop loop and start new game
            if done:
                break
        
        score = env.score_game()
        episode_scores.append(score)
        total_episode_score += score
        
        if ep % 10 == 0: # data tracking
            rewards.append(reward)
            scores.append(score)
        
        if ep % (episodes / num_logs) == 0: # logs
            
            print(f"Episode {ep}: epsilon={agent.epsilon:.3f}, avg_reward={episode_reward / (episodes / num_logs):.3f}, avg_game_score={total_episode_score/len(episode_scores)}")
            
            episode_reward = 0
            
            episode_scores = []
            total_episode_score = 0
    
    #agent.save("c:\\users\\benjaminsullivan\\downloads\\checkpoint3.npz")

if __name__ == "__main__":
    main()