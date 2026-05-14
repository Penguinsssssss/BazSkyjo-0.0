#BAZ Skyjo Testing Env

import random
import math
import numpy as np #type: ignore
seed = random.randint(0, 9999)
np.random.seed(seed)
print(f"Seed: {seed}")
#np.random.seed(4)
import copy
from collections import deque
from pandas import DataFrame
from pathlib import Path

# e
e = math.e

# activation functions
def sig(input): return 1 / (1 + np.exp(-input)) # sigmoid function
def relu(input): return np.maximum(0, input) # rectified linear
def expo(input): return np.exp(input) # exponent

# derivative activaiton functions
def d_relu(x): return (x > 0).astype(float) # rectified linear

# training settings
numtrials = 15000 # how many episodes before training is declared done?
isDebug = False # print 1 game?
base_path = Path(__file__).parent
AGENT_PATH = f"{base_path}\\checkpoint.npz" # create checkpoint file
isLoading = False # load already existing model to continue training?
SAVE_PATH = AGENT_PATH # save to another location?
isSaving = not isDebug # save progress each 1% of training?

class DuelingDQN:
    
    def __init__(self, state_size, action_size, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, tau=0.01):
        
        # consts/globals
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon # exploration chance
        self.epsilon_decay = epsilon_decay # exploration rate of change
        self.epsilon_min = epsilon_min # NN will always have a small chance to explore during training
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
        
        # apply backprop
        dQ = (2 * (Q_pred - target)) / Q_pred.shape[0]
        dV = np.sum(dQ, axis=1, keepdims=True)
        dA = dQ - np.mean(dQ, axis=1, keepdims=True)
        dA_prev = self.advantage.backward(dA)
        dV_prev = self.value.backward(dV)
        dTrunk = dA_prev + dV_prev
        for layer in reversed(self.layers):
            dTrunk = layer.backward(dTrunk)
        
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
        
        # clip gradients
        np.clip(self.dW, -1, 1, out=self.dW)
        np.clip(self.db, -1, 1, out=self.db)
        
        # update values
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

class Skyjo_Env(Environment):
    
    def __init__(self, debug=False):
        
        # 12 cards, discard pile, average value of unknown, lowest number of unknowns left, pendingcard, phase, which rows have a double, which rows have a triple, sum of hand, turns into game
        super().__init__(state_size=31, action_size=24)
        
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
        self.hand = [None] * 12
        # reveal two random cards
        for i in random.sample(range(12), 2):
            self.hand[i] = self.deck.pop()
        self.phand = copy.deepcopy(self.hand)
        
        # globals for phase
        self.phase = "main"
        self.pendingcard = None
        
        # check if game is over
        self.done = False
        
        # globals for bonus
        self.bonus = 0
        self.turn = 1
        
        return self.encode_state()
    
    def norm(self, card):
        if card is None: return 1.0 # sentinel
        else: return (card + 2) / 15 # norm value (exc. 1 for sentinel)
    
    def encode_state(self):
        
        # normalize all of own cards and deck avg
        own = [self.norm(i) for i in self.hand]
        avg = self.norm(sum(self.deck) / len(self.deck))
        
        # check each row for doubles
        doubles = []
        for i in range(4):
            col = [self.hand[i], self.hand[i + 4], self.hand[i + 8]]
            known = [c for c in col if c is not None]
            doubles.append(1 if len(known) != len(set(known)) else 0)
        
        triples = []
        
        # check each row for triple
        triples = []
        for i in range(4):
            col = [self.hand[i], self.hand[i + 4], self.hand[i + 8]]
            triples.append(1 if col[0] is not None and col[0] == col[1] == col[2] else 0)
        
        # gather sum of hand
        sums = []
        for i in range(4):
            col_sum = (
                (self.hand[i] if self.hand[i] is not None else avg) +
                (self.hand[i + 4] if self.hand[i + 4] is not None else avg) +
                (self.hand[i + 8] if self.hand[i + 8] is not None else avg)
            )
            sums.append((col_sum + 6) / 45) # norm
        
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
            sums +
            [self.turn / 100]
        )
    
    def step(self, action):
        
        # print hand for debug
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
        
        # main: 0 -> draw from deck | 1 -> take discard
        # pending: 0-11 pending -> hand | 12-23 pending -> discard + reveal one card in hand
        if self.phase == "main":
            
            if action == 0: # TAKE FROM DRAW
                
                # draw card from deck and query dqn
                
                self.pendingcard = self.deck.pop()
                
                self.dbg(f"Action 0 (draw) chosen (card {self.pendingcard})")
            
            else: # TAKE FROM DISCARD
                
                self.dbg(f"Action 1 chosen (take discard) discard is: {self.discard}")
                
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
            self.turn += 1
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
        
        # init bonus value
        bonus = 0
        
        # currently no reward for replacing an unknown
        # reward of old^2 - new^2 for replacing a card
        if wasUnknown:
            bonus += 0
        else:
            bonus += ((old*abs(old)) - (new*abs(new))) / 60
        
        # find out doubles + triples
        row = []
        for i in range(3):
            row.append(self.hand[(slot % 4) + (i * 4)])
        
        # for each row if created a double + 1 | if created a triple + 10
        if row.count(new) >= 2:
            bonus += 1
            if row.count(new) == 3:
                bonus += 10
        
        # penalize for high score, which gets more and more important as the game goes on
        #bonus += (20 - self.score_game()) * (self.turn / 20)
        
        return bonus
    
    def calcreward(self):
        
        # get deck avg
        avg = sum(self.deck) / len(self.deck)
        
        # prev hand expected value
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
        
        # current hand expected value
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
        
        # reward is the difference of prev hand - hand
        reward = pExpected - cExpected
        reward += self.bonus # add bonus
        
        # if last turn of game -> add game score to reward
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
    
    episodes = 1 if isDebug else numtrials
    max_steps = 1000
    
    # data collection settings
    episode_scores = []
    total_episode_score = 0
    episode_reward = 0
    rewards = []
    scores = []
    
    # visual indicator for impacient humans
    num_logs = numtrials/100
    
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
        #print(f"Game end {score}")
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