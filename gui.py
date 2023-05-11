import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

BOUNDRYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

MODEL = load_model('../MNIST_Digit_Classification_Model_with_GUI')

LABELS = {0: 'Zero', 1: 'One',
          2: 'Two', 3: 'Three',
          4: 'Four', 5: 'Five',
          6: 'Six', 7: 'Seven',
          8: 'Eight', 9: 'Nine'}

# Initializing the pygame module
pygame.init()

FONT = pygame.font.Font('freesansbold.ttf', 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []

IMAGESAVE = False
img_count = 1

PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:  # add this condition for detecting 'esc' key press
                pygame.quit()
                sys.exit()

            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x: rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f'image{img_count}.png', img_arr)
                img_count += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values= 0)
                image = cv2.resize(image, (28, 28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                textSurface = FONT.render(label, True, RED, WHITE)

                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                # Draw a red rectangle around the digit
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

                DISPLAYSURF.blit(textSurface, textRecObj)

    pygame.display.update()

