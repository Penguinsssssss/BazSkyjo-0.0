#skyjo botjo take 2

import random
import pygame
import time

class Deck():
    
    def __init__(self):
        
        self.deck = []
        for i in range(13):
            for _ in range(10):
                if i == 0: self.deck.append(-1)
                else: self.deck.append(i)
        for i in range(5): self.deck.append(-2)
        for i in range(15): self.deck.append(0)
        self.deck.sort()
        
        self.discard = self.drawCard()
    
    def drawCard(self):
        
        card = random.choice(self.deck)
        self.deck.remove(card)
        return card

class Player():
    
    def __init__(self, name, deck):
        
        self.name = name
        self.deck = deck
        self.hand = [
            [deck.drawCard(), None, None],
            [deck.drawCard(), None, None],
            [None, None, None],
            [None, None, None]
        ]
    
    def turn(self):
        
        self.card = None
        
        self.devDisplayHand()
        
        dAction = input(f"It is now {self.name}'s turn!\n(1): Take the discarded card [{self.deck.discard}]\n(2): Draw a random card [?]\nSelect your option: ")
        while dAction not in ("1", "2"):
            dAction = input(f"\nInvalid input, please try again!\n(1): Take the discarded card [{self.deck.discard}]\n(2): Draw a random card [?]\nSelect your option: ")
        
        if dAction == "1":
            self.card = self.deck.discard
            row = int(input(f"\nWhich row would you like to place that card?\n(1): {self.hand[0]}\n(2): {self.hand[1]}\n(3): {self.hand[2]}\n(4): {self.hand[3]}\n"))
        else:
            self.card = self.deck.drawCard()
            print(f"\nYou drew a [{self.card}]")
            row = int(input(f"\nWhich row would you like to place that card?\n(1): {self.hand[0]}\n(2): {self.hand[1]}\n(3): {self.hand[2]}\n(4): {self.hand[3]}\n(5): Discard\n"))
        
        if row != 5:
            collumn = int(input(f"\nWhich collumn would you like to place that card?\n(1): (2): (3): \n{self.hand[row - 1]}\n"))
            self.replace(self.card, [row - 1, collumn - 1])
        else:
            self.deck.discard = self.card
            row = int(input(f"\nIn which row would you like to reveal a card?\n(1): {self.hand[0]}\n(2): {self.hand[1]}\n(3): {self.hand[2]}\n(4): {self.hand[3]}\n"))
            collumn = int(input(f"\nIn which collumn would you like to reveal card?\n(1): (2): (3): \n{self.hand[row - 1]}\n"))
            self.reveal([row - 1, collumn - 1])
        
        print()
    
    def replace(self, card, d):
        
        if self.hand[d[0]][d[1]] is None:
            self.deck.discard = self.deck.drawCard()
        else:
            self.deck.discard = self.hand[d[0]][d[1]]
        self.hand[d[0]][d[1]] = card
    
    def reveal(self, d):
        self.hand[d[0]][d[1]] = self.deck.drawCard()
    
    def isGameOver(self):
        
        for i in self.hand:
            if None in i:
                return True
        return False
    
    def devDisplayHand(self):
        
        print(f"{self.name}'s hand: ")
        
        for i in self.hand:
            print(i)
        print()
    
    def score(self):
        score = 0
        for i in range(len(self.hand)):
            for j in range(len(self.hand[i])):
                if self.hand[i][j] is None: self.hand[i][j] = self.deck.drawCard()
        
        for i in self.hand:
            if i[0] == i[1] == i[2]: score += 0
            else: score += int(i[0]) + int(i[1]) + int(i[2])
        
        return score

class Game():
    
    def __init__(self, players):
        
        self.players = []
        self.deck = Deck()
        self.running = True
        
        for i in players:
            self.players.append(Player(i, self.deck))
    
    def turns(self):
        for i in self.players:
            i.turn()
        self.isGameOver()
    
    def isGameOver(self):
        for i in self.players:
            self.running = self.running and i.isGameOver()
    
    def devShowHands(self):
        for i in self.players: i.devDisplayHand()
    
    def scorePlayers(self):
        self.scores = []
        for i in self.players:
            print(i.hand, i.name)
            print(i.score())
            print(i.hand)

class Bot():
    
    def __init__(self):
        pass
    
    def evaluate(self, board):
        pass

class Display():
    
    def __init__(self, hand):
        
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption('Skyjo Botjo')
        
        self.hand = hand
    
    def drawCard(self, card, pos):
        if card is None: card = "?"
        xdim = 35
        ydim = 50
        colordict = {
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
            "?": "#999999"
        }
        pygame.draw.rect(self.screen, colordict[card], (pos[0], pos[1], xdim, ydim), 0, 5, 5)
        pygame.draw.rect(self.screen, "#000000", (pos[0], pos[1], xdim, ydim), 3, 5, 5)
        text = pygame.font.SysFont("Bahnschrift", 32).render(str(card), True, "#000000")
        textRect = text.get_rect()
        textRect.center = (pos[0] + (xdim / 2), pos[1] + (ydim / 2))
        self.screen.blit(text, textRect)
    
    def drawHand(self, hand):
        sy = 50
        for i in hand:
            sx = 50
            for j in i:
                self.drawCard(j, [sx, sy])
                sx += 45
            sy += 60
    
    def update(self):
        self.screen.fill("#ffffff")
        self.drawHand(self.hand)
        pygame.display.flip()

def main():
    game = Game(["Foo", "Baz"])
    """display = Display(game.players[0].hand)
    running = True
    while running:
        display.update()
        time.sleep(1)"""
    while game.running:
        game.turns()
    #game.devShowHands()
    #print(game.deck.deck)
    #print(game.deck.discard)
    print(game.scorePlayers())

main()