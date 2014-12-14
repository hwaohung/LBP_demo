import cv2
import numpy as np
import pylab
from matplotlib import pyplot as plt
import time

def flip_count(num):
    bits = "{0:08b}".format(num)
    flips = [(bits[i]!=bits[i+1]) for i in range(7)]
    return sum(flips)

# Input the code entry list(MSB->LSB)
def calc_value(code):
    value = 0
    for i in range(len(code)):
        value = (value<<1) + code[i]

    return value

# Input: Gray image
def gen_LBP(image):
    h, w = image.shape
    LBP = np.zeros((h-2, w-2), np.uint8)
    hist = [0 for i in range(59)]
    offsets = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,-1), (-1,0)] 
    for x in range(1, h-1):
        for y in range(1, w-1):
            value = 0
            for i, j in offsets:
                value = value << 1
                value += int(int(image[x+i, y+j])-int(image[x, y]) >= 0)

            LBP[x-1, y-1] = value
            hist[LBP_mapper[value]] += 1

    return LBP, hist

def gen_uniform_values(code):
    # Find uncertain bits's power flag
    flags = [len(code)-1-i for i in range(len(code)) if (code[i]) == -1]
    # Amount of uncertain bits permutation
    amount = 2 ** len(flags)

    # Code's min value
    base = sum([2**(len(code)-1-i) for i in range(len(code)) if (code[i]) == 1])

    uniform_values = list()
    bit_format = "{0:0" + str(len(flags)) + "b}"
    for i in range(amount):
        candidate = base
        bits = bit_format.format(i)
        for j in range(len(flags)):
            candidate += int(bits[j]) * (2**flags[j])

        if flip_count(candidate) <= 2:
            uniform_values.append(candidate)

    return uniform_values

def gen_NRLBP(image, threshold):
    h, w = image.shape
    offsets = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,-1), (-1,0)]
    hist = [0 for i in range(59)]
    for x in range(1, h-1):
        print x
        for y in range(1, w-1):
            code = list()
            for i, j in offsets:
                Zp = int(image[x+i, y+j])-int(image[x, y])
                if Zp >= threshold:
                    code.append(1)
                elif Zp <= -threshold:
                    code.append(0)
                # Uncertain bit
                else:
                    code.append(-1)
            
            if code.count(-1) != 0:
                candidates = gen_uniform_values(code)
                for candidate in candidates:
                    slot = LBP_mapper[candidate]
                    hist[slot] += 1.0/len(candidates)
            else:
                slot = LBP_mapper[calc_value(code)]
                hist[slot] += 1
    
    return hist

def gen_histogram_entries(image):
    h, w = image.shape
    entries = list()
    for x in range(h):
        for y in range(w):
            entries.append(LBP_mapper[image[x, y]])
    return entries

def draw_histogram(X, Y):
    plt.bar(X, Y, width=0.9, linewidth=1)
    plt.show()

def show_histogram(values):
    bins = range(min(values), max(values)+1)
    pylab.hist(values, bins, align="mid")
    pylab.show()


if __name__ == "__main__":
    # LBP value => bin ID
    bin_id = 0
    LBP_mapper = [0 for i in range(256)]
    for i in range(256):
        # Uniform
        if flip_count(i) <= 2:
            LBP_mapper[i] = bin_id
            bin_id += 1
        # Non-uniform
        else:
            LBP_mapper[i] = 58

    filename = "lena.bmp"
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    t = time.time() 
    #result_image, hist = gen_LBP(gray_image)
    hist = gen_NRLBP(gray_image, 5)
    draw_histogram(range(0, 59), hist)
    #entries = gen_histogram_entries(LBP_image)
    #show_histogram(entries)
    print "Cost: {0}".format(time.time()-t)
