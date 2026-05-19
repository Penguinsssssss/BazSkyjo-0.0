import pygame

def drawTexts(screen, sTexts):
    for i in sTexts:
        text = i[0].render(i[1], True, i[2]) #font, text, color, centred on, pos(x,y)
        textRect = text.get_rect()
        if i[3] == "c":
            textRect.center = i[4]
        elif i[3] == "l":
            textRect.left = i[4][0]
            textRect.centery = i[4][1]
        elif i[3] == "lb":
            textRect.left = i[4][0]
            textRect.bottom = i[4][1]
        elif i[3] == "lt":
            textRect.left = i[4][0]
            textRect.top = i[4][1]
        elif i[3] == "r":
            textRect.right = i[4][0]
            textRect.centery = i[4][1]
        elif i[3] == "rb":
            textRect.right = i[4][0]
            textRect.bottom = i[4][1]
        elif i[3] == "rt":
            textRect.right = i[4][0]
            textRect.top = i[4][1]
        screen.blit(text, textRect)

def elo(pElo, eElo, wl, k=32):
    playerExpected = 1 / (1 + 10 ** ((eElo - pElo) / 400))
    playerActual = 1 if wl else 0
    player += k * (playerActual - playerExpected)
    return round(player)

def drawAbsolute(x, y, ex, ey, scx, scy): return (scx * x, scy * y, scx * (ex - x), scy * (ey - y))

def text(screen, text, pos, color, size, font="Arial.ttf", center=None):
    if font.lower().endswith(".ttf"):
        tfont = pygame.font.Font(font, size)
    else:
        tfont = pygame.font.SysFont(font, size)

    ttext = tfont.render(text, True, color)
    textRect = ttext.get_rect()
    
    if center == None or center == "center":
        textRect.center = pos
    elif center == "left":
        textRect.left = pos[0]
        textRect.centery = pos[1]
    elif center == "right":
        textRect.right = pos[0]
        textRect.centery = pos[1]
    elif center == "top":
        textRect.centerx = pos[0]
        textRect.top = pos[1]
    elif center == "bottom":
        textRect.centerx = pos[0]
        textRect.bottom = pos[1]
    elif center == "topleft":
        textRect.left = pos[0]
        textRect.top = pos[1]
    elif center == "bottomleft":
        textRect.left = pos[0]
        textRect.bottom = pos[1]
    elif center == "topright":
        textRect.right = pos[0]
        textRect.top = pos[1]
    elif center == "bottomright":
        textRect.right = pos[0]
        textRect.bottom = pos[1]
    screen.blit(ttext, textRect)

def normalize(inputs):
    base, norm = sum(inputs), []
    for i in inputs: norm.append(i / base)
    return norm