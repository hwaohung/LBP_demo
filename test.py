import cv2
import numpy as np
import pylab
import time

def flip_count(num):
    bits = "{0:08b}".format(num)
    flips = [(bits[i]!=bits[i+1]) for i in range(7)]
    return sum(flips)

# Input the code entry list(MSB->LSB)
def calc_value(code):
    value = 0
    for i in range(code):
        value = (value<<1) + code[i]

    return value

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

def gen_uniform_codes(code):
    # Find uncertain bits's power flag
    flags = [len(code)-1-i for i in range(len(code)) if (code[i]) == -1]
    # Amount of uncertain bits permutation
    amount = 2 ** len(flags)

    # Code's min value
    base = sum([2**(len(code)-1-i) for i in range(len(code)) if (code[i]) == 1])
    bit_format = "{0:0" + uncertain_bits + "b}"
    for i in range(amount):
        candidate = base
        bits = bit_format.format(i)
        for j in range(len(flags)):
            candidate += bits[i] * (2**flag[j])

        if flip_count(candidate) <= 2:
            uniform_codes.append(candidate)

    return uniform_codes

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
                candidates = gen_uniform_codes(code)
                for candidate in candidates:
                    hist[calc_value(candidate)] += 1.0/len(candidates)
            else:
                hist[calc_value(code)] += 1
    
    return hist

def gen_histogram_entries(image):
    h, w = image.shape
    entries = list()
    for x in range(h):
        for y in range(w):
            entries.append(LBP_mapper[image[x, y]])
    return entries

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
            LBP_mapper[i] = 59

    filename = "lena.bmp"
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    t = time.time() 
    LBP_image = gen_LBP(gray_image)
    entries = gen_histogram_entries(LBP_image)
    show_histogram(entries)
    print "Cost: {0}".format(time.time()-t)
