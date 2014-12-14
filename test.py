import cv2
import numpy as np
import pylab
from matplotlib import pyplot as plt
import time

def flip_count(num):
    bits = "{0:08b}".format(num)
    flips = 0
    for i in range(7):
        flips += (bits[i] is not bits[i+1])

    return flips

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

# Input the code entry list(MSB->LSB)
def calc_value(code):
    value = 0
    for i in range(len(code)):
        value = (value<<1) + code[i]

    return value

def gen_uniform_values(code):
    # Find uncertain bits's power flag
    flags = list()
    # Code's min value
    base = 0 

    bit_len = len(code)
    for i in range(bit_len):
        base <<= 1
        if code[i] == -1:
            flags.append(bit_len-1-i)
        else:
            base += code[i]

    # Amount of uncertain bits permutation
    amount = 2 ** len(flags)

    uniform_values = list()
    bit_format = "{0:0" + str(len(flags)) + "b}"
    for i in range(amount):
        candidate = base
        bits = bit_format.format(i)
        for j in range(len(flags)):
            if bits[j] is "1":
                candidate += (2**flags[j])

        if flip_count(candidate) <= 2:
            uniform_values.append(candidate)

    return uniform_values

def gen_NRLBP(image, threshold):
    h, w = image.shape
    offsets = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,-1), (-1,0)]
    hist = [0 for i in range(59)]
    for x in range(1, h-1):
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
                values = gen_uniform_values(code)
                if len(values) == 0:
                    hist[len(hist)-1] += 1
                else:
                    for value in values:
                        slot = LBP_mapper[value]
                        hist[slot] += 1.0/len(values)
            else:
                slot = LBP_mapper[calc_value(code)]
                hist[slot] += 1

    return hist

def draw_histogram(X, Y, filename):
    plt.bar(X, Y, width=0.9, linewidth=1)
    plt.savefig(filename)
    plt.clf()
    #plt.show()


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

    filename = "lena.jpg"
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    t = time.time() 
    result_image, hist = gen_LBP(gray_image)
    draw_histogram(range(0, 59), hist, "LBP_hist.png")
    print sum(hist), hist[len(hist)-1]

    hist = gen_NRLBP(gray_image, 2)
    draw_histogram(range(0, 59), hist, "NRLBP_hist.png")
    print sum(hist), hist[len(hist)-1]
    print "Cost: {0}".format(time.time()-t)
