import numpy as np
import cv2

# Shorter code to generate checkerboard

# number of corners
VERT = 11
HORI = 17

ROWS = VERT + 1
COLS = HORI + 1
BLOCKSIZE = 60
MARGIN = 180

HEIGHT = ROWS*BLOCKSIZE + 2*MARGIN
WIDTH = COLS*BLOCKSIZE + 2*420

img = np.ones((HEIGHT, WIDTH), np.uint8)*255

for r in range(ROWS):
    for c in range(COLS):
        if (r+c) % 2 == 1:
            x = 420 + BLOCKSIZE*c
            y = MARGIN + BLOCKSIZE*r
            img[y:y+BLOCKSIZE, x:x+BLOCKSIZE] = 0

cv2.imwrite('chessboard_{0}x{1}.png'.format(VERT, HORI), img)
print("done")