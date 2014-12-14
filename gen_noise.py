import cv2
import sys
import numpy as np


def noise_embed(image, mu, sigma): 
    h, w = image.shape
    noises = np.random.normal(mu, sigma, h*w)
    noise_image = image + noises.reshape(h, w)

    for i in range(h):
        for j in range(w):
            if noise_image[i, j] > 255.0:
                noise_image[i, j] = 255.0

            elif noise_image[i, j] < 0.0:
                noise_image[i, j] = 0.0

    return noise_image.round()


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray_image.shape
    noises = np.random.normal(0.0, 0.1, h*w)
    noise_image = gray_image + noises.reshape(h, w)

    for i in range(h):
        for j in range(w):
            if noise_image[i, j] > 255:
                noise_image[i, j] = 255

            elif noise_image[i, j] < 0:
                noise_image[i, j] = 0

    noise_image.ground()


    noises = gray_image - noise_image

    print noise_image.min(), noise_image.max()
    print noises.var() ** 0.5

    cv2.imwrite("noise.png", noise_image)
