import cv2
import numpy as np
import pylab
import time



def flip_count(num):
    bits = "{0:08b}".format(num)
    flips = [(bits[i]!=bits[i+1]) for i in range(7)]
    return sum(flips)

# Input: Gray image
def gen_LBP(image):
    h, w = image.shape
    LBP = np.zeros((h-2, w-2), np.uint8)

    offsets = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,-1), (-1,0)] 
    for x in range(1, h-1):
        for y in range(1, w-1):
            value = 0
            for i, j in offsets:
                value = value << 1
                value += int(int(image[x+i, y+j])-int(image[x, y]) >= 0)

            LBP[x-1, y-1] = value

    return LBP


def draw_histogram(image):
    h, w = image.shape
    entries = list()
    for x in range(h):
        for y in range(w):
            bits = "{0:08b}".format(image[x, y])
            flips = [(bits[i]!=bits[i+1]) for i in range(7)]
            # Non-uniform
            if sum(flips) > 2:
                entries.append(58)
            # Uniform
            else:
                entries.append(LBP_mapper[image[x, y]])

    show_histogram(entries)


def show_histogram(values):
    bins = range(min(values), max(values)+1)
    pylab.hist(values, bins, align='mid')
    pylab.show()


if __name__ == "__main__":
    # LBP value => bin ID
    LBP_bins = [i for i in range(256) if flip_count(i) <= 2]
    LBP_mapper = {LBP_bins[i]: i for i in range(len(LBP_bins))}

    filename = "lena.bmp"
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    t = time.time() 
    LBP_image = gen_LBP(gray_image)
    draw_histogram(LBP_image)
    print "Cost: {0}".format(time.time()-t)
    #print gray_image
    #cv2.imwrite("test.png", gray_image)
