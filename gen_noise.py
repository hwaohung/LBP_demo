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
