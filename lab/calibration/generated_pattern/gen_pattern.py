#!/usr/bin/env python

"""gen_pattern.py
Usage example:
python gen_pattern.py -o out.svg -r 11 -c 8 -T circles -s 20.0 -R 5.0 -u mm -w 216 -h 279
-o, --output - output file (default out.svg)
-r, --rows - pattern rows (default 11)
-c, --columns - pattern columns (default 8)
-T, --type - type of pattern, circles, acircles, checkerboard (default circles)
-s, --square_size - size of squares in pattern (default 20.0)
-R, --radius_rate - circles_radius = square_size/radius_rate (default 5.0)
-u, --units - mm, inches, px, m (default mm)
-w, --page_width - page width in units (default 216)
-h, --page_height - page height in units (default 279)
-a, --page_size - page size (default A4), supersedes -h -w arguments
-H, --help - show help
"""

import argparse

from generated_pattern.svgfig import *
import cv2
import numpy as np
import cairosvg as cairosvg
                                                                                                                                                                  
class PatternMaker:
    def __init__(self, cols, rows, output, units, square_size, page_width, page_height):
        self.cols = cols
        self.rows = rows
        self.output = output
        self.units = units
        self.square_size = square_size
        self.width = page_width
        self.height = page_height
        self.g = SVG("g")  # the svg group container

    def make_checkerboard_pattern(self):
        spacing = self.square_size
        xspacing = (self.width - self.cols * self.square_size) / 2.0
        yspacing = (self.height - self.rows * self.square_size) / 2.0
        for x in range(0, self.cols):
            for y in range(0, self.rows): 
                if x % 2 == y % 2:
                    square = SVG("rect", x=x * spacing + xspacing, y=y * spacing + yspacing, width=spacing,
                                 height=spacing, fill="black", stroke="none")
                    self.g.append(square)

    def save_2d_projector_corner_cordinates(self):
        checkerboardcordinates = []
        xspacing = (self.width - self.cols * self.square_size) / 2.0
        yspacing = (self.height - self.rows * self.square_size) / 2.0
        for y in range(int(yspacing),self.height-int(yspacing), self.square_size):
            if (y + self.square_size > self.height -int(yspacing)):
                break
            for x in range(int(xspacing), self.width-int(xspacing), self.square_size):
                if (x + self.square_size > self.width -int(xspacing)):
                    break
                if(x > int(xspacing) and y > int(yspacing)):   
                    checkerboardcordinates = np.append(checkerboardcordinates, np.array([x,y]))
        checkerboardcordinates = np.reshape(checkerboardcordinates, (-1,2))
        #print(checkerboardcordinates)
        #print(len(checkerboardcordinates))
        return checkerboardcordinates

    def save(self):
        c = canvas(self.g, width="%d%s" % (self.width, self.units), height="%d%s" % (self.height, self.units),
                   viewBox="0 0 %d %d" % (self.width, self.height))
        c.save(self.output)


def main():
    # parse command line options
   
    output = r"C:\Users\eivin\Desktop\NTNU master-PUMA-2019-2021\3.studhalvår\TPK4560-Specalization-Project\projector-calibration\generated_pattern\chessboard.svg"
    columns = 10
    rows = 7
    p_type = "checkerboard"
    units = "px"
    square_size = 90
    page_size = "CUSTOM"
    # page size dict (ISO standard, mm) for easy lookup. format - size: [width, height]
    page_sizes = {"CUSTOM": [1024, 768], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297],
                  "A5": [148, 210]}
    page_width = page_sizes[page_size.upper()][0]
    page_height = page_sizes[page_size.upper()][1]
    pm = PatternMaker(columns, rows, output, units, square_size, page_width, page_height)
    # dict for easy lookup of pattern type
    mp = {"checkerboard": pm.make_checkerboard_pattern}
    mp[p_type]()
    # this should save pattern to output
    pm.save()
    cooord = pm.save_2d_projector_corner_cordinates()
    #print(cooord)
    #convert from vector file(svg) to png
    cairosvg.svg2png(url='chessboard.svg', write_to='image.png')

    img2 = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)
    if img2.shape[2] == 4:     # we have an alpha channel
        a1 = ~img2[:,:,3]        # extract and invert that alpha
        img2 = cv2.add(cv2.merge([a1,a1,a1,a1]), img2)   # add up values (with clipping)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)    # strip alpha channel

    cv2.imshow('Checkerboard', img2); cv2.waitKey(0); cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

    

    


    
