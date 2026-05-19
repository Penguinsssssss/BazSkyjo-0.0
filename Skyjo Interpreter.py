# Skyjo.py (~1/3 of this is ai generated, as I ran out of time by the end of the year)

from pathlib import Path
import random
import sys
import time
import numpy as np  # type: ignore
import pygame  # type: ignore
import penguinsmodule as pm  # type: ignore
import tkinter as tk
root = tk.Tk()
root.withdraw()
scX = int(root.winfo_screenwidth() * 0.5)
scY = int(root.winfo_screenheight() * 0.5)
root.destroy()

BASE_PATH = Path(__file__).parent
AGENT_PATH = BASE_PATH / "checkpoint 512x3 debug.npz"
STATE_SIZE = 31
DEFAULT_ACTION_SIZE = 24  # 0-1 draw/pickup discard, 0-11 = place pending, 12-23 = discard pending and reveal slot 0-11
BOT_PAUSE_SECONDS = 1.25

CARD_COLORS = {
    -2: "#6a70a2",
    -1: "#6a70a2",
    0: "#8ab7ce",
    1: "#73a252",
    2: "#73a252",
    3: "#73a252",
    4: "#73a252",
    5: "#e3cc4a",
    6: "#e3cc4a",
    7: "#e3cc4a",
    8: "#e3cc4a",
    9: "#b5463d",
    10: "#b5463d",
    11: "#b5463d",
    12: "#b5463d",
    None: "#999999",
    "outline": "#000000",
    "selected": "#55FF55",
}

def relu(x):return np.maximum(0, x)
def rect_scaled(x, y, ex, ey): return pygame.Rect(pm.drawAbsolute(x, y, ex, ey, scX, scY))

class DuelingDQN:
    
    # only forward pass
    
    def __init__(self, state_size=STATE_SIZE, action_size=DEFAULT_ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.model = Network([state_size, 512, 512, 512, 64], action_size)
    
    @classmethod
    def from_checkpoint(cls, path):
        path = Path(path)
        if not path.exists():
            print(f"No checkpoint found at {path}. Bot will use random initialized weights.")
            return cls()
        
        ckpt = np.load(path, allow_pickle=False)
        state_size = int(ckpt["state_size"][0]) if "state_size" in ckpt else STATE_SIZE
        action_size = int(ckpt["action_size"][0]) if "action_size" in ckpt else DEFAULT_ACTION_SIZE
        agent = cls(state_size, action_size)
        agent.load(path)
        print(f"Loaded bot checkpoint: {path}")
        return agent
    
    def load(self, path):
        
        ckpt = np.load(path, allow_pickle=False)
        
        for i, layer in enumerate(self.model.layers):
            layer.weights = ckpt[f"m_layers_{i}_W"]
            layer.biases = ckpt[f"m_layers_{i}_b"]
            
        self.model.value.weights = ckpt["m_value_W"]
        self.model.value.biases = ckpt["m_value_b"]
        self.model.advantage.weights = ckpt["m_adv_W"]
        self.model.advantage.biases = ckpt["m_adv_b"]
    
    def choose_action(self, state, legal_actions, return_q=False):
        
        q_values = self.model.calculate(state).flatten()
        masked_q = np.full_like(q_values, -1e9, dtype=np.float64)
        
        legal_actions = [a for a in legal_actions if 0 <= a < len(q_values)]
        if not legal_actions:
            raise ValueError("No legal actions are inside the model action range.")
        
        masked_q[legal_actions] = q_values[legal_actions]
        action = int(np.argmax(masked_q))
        if return_q:
            return action, masked_q
        return action

class Network:
    
    # only forward pass
    
    def __init__(self, layer_structure, num_actions):
        self.layers = [
            Layer(layer_structure[i], layer_structure[i + 1], "relu")
            for i in range(len(layer_structure) - 1)
        ]
        self.value = Layer(layer_structure[-1], 1, "linear")
        self.advantage = Layer(layer_structure[-1], num_actions, "linear")
        
    def calculate(self, inputs):
        
        # return q vals
        
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            
        out = inputs
        for layer in self.layers:
            out = layer.calculate(out)
            
        value = self.value.calculate(out)
        advantage = self.advantage.calculate(out)
        return value + (advantage - np.mean(advantage, axis=1, keepdims=True))

class Layer:
    
    # only forward pass
    
    def __init__(self, num_inputs, num_neurons, activation):
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.activation = activation
    
    def calculate(self, inputs):
        out = np.dot(inputs, self.weights) + self.biases
        if self.activation == "relu":
            return relu(out)
        return out

class Deck:
    
    # skyjo deck is 10 of each card with 15 0s and 5 -2s
    
    def __init__(self):
        self.deck = []
        for i in range(13):
            # Original code made i == 0 represent -1 cards.
            self.deck.extend([-1 if i == 0 else i] * 10)
        self.deck.extend([-2] * 5)
        self.deck.extend([0] * 15)
        random.shuffle(self.deck)
        self.discard = self.draw_card()
        
    def draw_card(self):
        if not self.deck:
            raise RuntimeError("The draw pile is empty.") # ty chat
        return self.deck.pop()
    
    def get_avg(self):
        return sum(self.deck) / len(self.deck) if self.deck else 0

class Game:
    
    def __init__(self, manager, agent):
        self.manager = manager
        self.agent = agent
        self.deck = Deck()
        self.manager.deck = self.deck
        self.manager.game = self
        self.manager.latest_move = None
        self.manager.pending_card = None
        self.manager.highlight_unknowns_player = None
        
        self.manager.hands = []
        for _ in range(2): # only two player support rn ;(((
            hand = [self.deck.draw_card(), self.deck.draw_card()] + [None] * 10
            self.manager.hands.append(hand)
        
        self.turns = 0
        self.game_over = False
    
    def norm(self, card):
        # norm card values
        if card is None:
            return 1.0
        return (card + 2) / 15
    
    def encode_state(self, hand, phase, pending_card):
        own = [self.norm(card) for card in hand]
        deck_avg_raw = self.deck.get_avg()
        deck_avg_norm = self.norm(deck_avg_raw)
        
        doubles = []
        triples = []
        sums = []
        for i in range(4):
            col = [hand[i], hand[i + 4], hand[i + 8]]
            known = [c for c in col if c is not None]
            doubles.append(1 if len(known) != len(set(known)) else 0)
            triples.append(1 if col[0] is not None and col[0] == col[1] == col[2] else 0)
            
            col_sum = sum(card if card is not None else deck_avg_raw for card in col)
            sums.append((col_sum + 6) / 45)
            
        lowest_unknowns = min(sum(card is None for card in hand) for hand in self.manager.hands)
        
        return np.array(
            own
            + [self.norm(self.deck.discard)]
            + [2 / 6]
            + [deck_avg_norm]
            + [lowest_unknowns / 12]
            + [self.norm(pending_card) if phase == "pending" else 1.0]
            + [1 if phase == "pending" else 0]
            + doubles
            + triples
            + sums
            + [self.turns / 100],
            dtype=np.float32,
        )
    
    def unknown_slots(self, player_idx): return [i for i, card in enumerate(self.manager.hands[player_idx]) if card is None]
    
    def score_hand(self, hand, reveal_unknowns=False):
        visible = [self.deck.get_avg() if card is None and reveal_unknowns else card for card in hand]
        score = 0
        for i in range(4):
            col = [visible[i], visible[i + 4], visible[i + 8]]
            if None in col:
                score += sum(card for card in col if card is not None)
            elif col[0] == col[1] == col[2]:
                score += 0
            else:
                score += sum(col)
        return round(score, 1)
    
    def draw_safe_reveal_card(self):
        if self.deck.deck:
            return self.deck.draw_card()
        return self.deck.discard
    
    def reveal_all_unknowns(self):
        for player_idx, hand in enumerate(self.manager.hands):
            for slot, card in enumerate(hand):
                if card is None:
                    hand[slot] = self.draw_safe_reveal_card()
        self.manager.highlight_unknowns_player = None
        self.manager.pending_card = None
    
    def finish_if_needed(self):
        # end game when one player is out of unknowns
        if any(all(card is not None for card in hand) for hand in self.manager.hands):
            self.reveal_all_unknowns()
            self.game_over = True
            p_score = self.score_hand(self.manager.hands[0])
            b_score = self.score_hand(self.manager.hands[1])
            if p_score < b_score:
                result = "Player wins"
            elif b_score < p_score:
                result = "Baz wins"
            else:
                result = "Tie"
            print(f"Game over | Your score: {p_score}, Baz score: {b_score} | {result} | press r to restart.")
            return True
        return False
    
    def place_pending_card(self, player_idx, slot, pending_card):
        old_card = self.manager.hands[player_idx][slot]
        self.manager.hands[player_idx][slot] = pending_card
        if old_card is not None:
            self.deck.discard = old_card
        else:
            self.deck.discard = self.draw_safe_reveal_card()
        
        self.manager.latest_move = {
            "player": player_idx,
            "slot": slot,
            "pile": None,
            "kind": "place",
        }
    
    def discard_pending_and_reveal(self, player_idx, reveal_slot, pending_card):
        if self.manager.hands[player_idx][reveal_slot] is not None:
            return
        
        self.deck.discard = pending_card
        self.manager.hands[player_idx][reveal_slot] = self.deck.draw_card()
        self.manager.latest_move = {
            "player": player_idx,
            "slot": reveal_slot,
            "pile": "discard",
            "kind": "discard_reveal",
        }
    
    def human_turn(self):
        self.manager.latest_move = None
        source = self.manager.wait_for_pile_click()
        
        if source == "discard":
            pending_card = self.deck.discard
        else:
            pending_card = self.deck.draw_card()
        
        self.manager.pending_card = pending_card
        self.manager.latest_move = {"pile": source, "player": None, "slot": None, "kind": "take"}
        
        choice = self.manager.wait_for_pending_choice(player_idx=0)
        if choice[0] == "place":
            self.place_pending_card(0, choice[1], pending_card)
        else:
            # Once the player clicks the discard pile, show the pending card there immediately.
            # Then only unknown cards are valid reveal targets, so highlight them while waiting.
            self.deck.discard = pending_card
            self.manager.pending_card = None
            self.manager.highlight_unknowns_player = 0
            self.manager.latest_move = {
                "player": None,
                "slot": None,
                "pile": "discard",
                "kind": "waiting_for_reveal",
            }
            reveal_slot = self.manager.wait_for_reveal_click(player_idx=0)
            self.manager.highlight_unknowns_player = None
            self.manager.hands[0][reveal_slot] = self.deck.draw_card()
            self.manager.latest_move = {
                "player": 0,
                "slot": reveal_slot,
                "pile": "discard",
                "kind": "discard_reveal",
            }
            revealed = self.manager.hands[0][reveal_slot]
        
        self.manager.highlight_unknowns_player = None
        self.manager.pending_card = None
        self.turns += 1
        self.finish_if_needed()
    
    def bot_turn(self):
        if self.game_over:
            return
        
        hand = self.manager.hands[1]
        self.manager.latest_move = None
        self.manager.wait_seconds(BOT_PAUSE_SECONDS)
        
        # main phase action mapping
        # 0 = draw, 1 = take discard
        main_state = self.encode_state(hand, "main", 1)
        main_action, main_q = self.agent.choose_action(main_state, [0, 1], return_q=True)
        
        if main_action == 1:
            pending_card = self.deck.discard
            source = "discard"
        else:
            pending_card = self.deck.draw_card()
            source = "draw"
            
        self.manager.pending_card = pending_card
        self.manager.latest_move = {"pile": source, "player": None, "slot": None, "kind": "take"}
        self.manager.wait_seconds(BOT_PAUSE_SECONDS)
        
        # pending phase action mapping
        # 0-11 = place pending card into slot
        # 12-23 = discard pending card, then reveal slot action - 12
        pending_state = self.encode_state(hand, "pending", pending_card)
        legal_place_actions = list(range(12))
        legal_reveal_actions = [12 + slot for slot in self.unknown_slots(1)]
        legal_pending_actions = legal_place_actions + legal_reveal_actions
        
        pending_action, slot_q = self.agent.choose_action(pending_state, legal_pending_actions, return_q=True)
        
        if 0 <= pending_action <= 11:
            self.place_pending_card(1, pending_action, pending_card)
        else:
            reveal_slot = pending_action - 12
            self.discard_pending_and_reveal(1, reveal_slot, pending_card)
            revealed = self.manager.hands[1][reveal_slot]
            
        self.manager.pending_card = None
        self.turns += 1
        print("Main Q:", np.round(main_q[:2], 3))
        print("Pending Q:", np.round(slot_q[: min(24, len(slot_q))], 3))
        self.manager.wait_seconds(BOT_PAUSE_SECONDS)
        self.finish_if_needed()
    
    def run_round(self):
        self.human_turn()
        if not self.game_over:
            self.bot_turn()

class PGManager:
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((scX, scY))
        pygame.display.set_caption("Skyjo")
        self.fill_color = "#FFFFFF"
        
        self.xlen = 0.075
        self.ylen = 0.15
        self.spacing = 0.025
        
        self.player_hand_pos = [0.025, 0.24]
        self.bot_hand_pos = [0.6, 0.24]
        
        self.font = pygame.font.SysFont("Arial", 32)
        self.small_font = pygame.font.SysFont("Arial", 20)
        self.pending_card = None
        self.deck = None
        self.hands = []
        self.game = None
        self.latest_move = None
        self.highlight_unknowns_player = None
    
    def tick(self):
        self.screen.fill(self.fill_color)
        
        if self.hands:
            self.draw_hand(self.hands[0], self.player_hand_pos, "You", player_idx=0)
            self.draw_hand(self.hands[1], self.bot_hand_pos, "Baz (AI)", player_idx=1)
        if self.deck:
            self.draw_piles()
        if self.pending_card is not None:
            self.draw_pending()
            
        pygame.display.flip()
    
    def draw_hand(self, hand, pos, label, player_idx):
        label_surf = self.small_font.render(label, True, "#000000")
        self.screen.blit(label_surf, (round(pos[0] * scX), round((pos[1] - 0.08) * scY)))
        
        for idx, value in enumerate(hand):
            x = idx % 4
            y = idx // 4
            selected = self.is_selected_card(player_idx, idx)
            card = Card(self.screen, self.font, value, self.xlen, self.ylen, selected=selected)
            card.draw([pos[0] + x * (self.spacing + self.xlen), pos[1] + y * (self.spacing + self.ylen)])
    
    def draw_piles(self):
        draw_selected = self.is_selected_pile("draw")
        discard_selected = self.is_selected_pile("discard")
        
        Card(self.screen, self.font, None, self.xlen, self.ylen, selected=draw_selected).draw(
            [0.5 - self.xlen / 2, 0.18]
        )
        Card(self.screen, self.font, self.deck.discard, self.xlen, self.ylen, selected=discard_selected).draw(
            [0.5 - self.xlen / 2, 0.43]
        )
        
        draw_label = self.small_font.render("Draw", True, "#000000")
        discard_label = self.small_font.render("Discard", True, "#000000")
        self.screen.blit(draw_label, (round((0.5 - self.xlen / 2) * scX), round(0.10 * scY)))
        self.screen.blit(discard_label, (round((0.5 - self.xlen / 2) * scX), round(0.35 * scY)))
    
    def draw_pending(self):
        Card(self.screen, self.font, self.pending_card, self.xlen, self.ylen, selected=True).draw(
            [0.5 - self.xlen / 2, 0.64]
        )
        label = self.small_font.render("Pending", True, "#000000")
        self.screen.blit(label, (round((0.5 - self.xlen / 2) * scX), round(0.60 * scY)))
    
    def is_selected_card(self, player_idx, slot):
        if (
            self.highlight_unknowns_player == player_idx
            and self.hands
            and self.hands[player_idx][slot] is None
        ):
            return True
        
        return bool(
            self.latest_move
            and self.latest_move.get("player") == player_idx
            and self.latest_move.get("slot") == slot
        )
    
    def is_selected_pile(self, pile):
        return bool(self.latest_move and self.latest_move.get("pile") == pile)
    
    def hand_rects(self, player_idx):
        base = self.player_hand_pos if player_idx == 0 else self.bot_hand_pos
        rects = []
        for idx in range(12):
            x = idx % 4
            y = idx // 4
            left = base[0] + x * (self.spacing + self.xlen)
            top = base[1] + y * (self.spacing + self.ylen)
            rects.append(rect_scaled(left, top, left + self.xlen, top + self.ylen))
        return rects
    
    def pile_rects(self):
        draw = rect_scaled(0.5 - self.xlen / 2, 0.14, 0.5 + self.xlen / 2, 0.14 + self.ylen)
        discard = rect_scaled(0.5 - self.xlen / 2, 0.39, 0.5 + self.xlen / 2, 0.39 + self.ylen)
        return draw, discard
    
    def wait_for_pile_click(self):
        while True:
            self.tick()
            for event in pygame.event.get():
                self.handle_global_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    draw_rect, discard_rect = self.pile_rects()
                    if draw_rect.collidepoint(event.pos):
                        return "draw"
                    if discard_rect.collidepoint(event.pos):
                        return "discard"
    
    def wait_for_pending_choice(self, player_idx):
        hand_rects = self.hand_rects(player_idx)
        while True:
            self.tick()
            for event in pygame.event.get():
                self.handle_global_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    _, discard_rect = self.pile_rects()
                    if discard_rect.collidepoint(event.pos):
                        return ("discard", None)
                    for idx, rect in enumerate(hand_rects):
                        if rect.collidepoint(event.pos):
                            return ("place", idx)
    
    def wait_for_reveal_click(self, player_idx):
        hand_rects = self.hand_rects(player_idx)
        while True:
            self.tick()
            for event in pygame.event.get():
                self.handle_global_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for idx, rect in enumerate(hand_rects):
                        if rect.collidepoint(event.pos):
                            if self.hands[player_idx][idx] is None:
                                return idx
    
    def wait_seconds(self, seconds):
        start = time.time()
        clock = pygame.time.Clock()
        while time.time() - start < seconds:
            clock.tick(60)
            for event in pygame.event.get():
                self.handle_global_event(event)
            self.tick()
    
    def handle_global_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and self.game:
            self.game.__init__(self, self.game.agent)

class Card:
    
    def __init__(self, screen, font, value, xlen, ylen, selected=False):
        self.screen = screen
        self.font = font
        self.value = value
        self.xlen = xlen
        self.ylen = ylen
        self.selected = selected
    
    def draw(self, pos):
        rect = rect_scaled(pos[0], pos[1], pos[0] + self.xlen, pos[1] + self.ylen)
        pygame.draw.rect(self.screen, CARD_COLORS[self.value], rect, 0, 3)
        
        outline_color = CARD_COLORS["selected"] if self.selected else CARD_COLORS["outline"]
        outline_width = 5 if self.selected else 3
        pygame.draw.rect(self.screen, outline_color, rect, outline_width, 3)
        
        text = str(self.value) if self.value is not None else "?"
        text_surf = self.font.render(text, True, "#000000")
        self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

def main():
    agent = DuelingDQN.from_checkpoint(AGENT_PATH)
    manager = PGManager()
    game = Game(manager, agent)
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(60)
        manager.tick()
        
        if game.game_over:
            for event in pygame.event.get():
                manager.handle_global_event(event)
            continue
        
        game.run_round()

if __name__ == "__main__":
    main()