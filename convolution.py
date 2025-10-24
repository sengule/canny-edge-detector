import numpy as np

def convolve2d(img, kernel):
    rows, cols = img.shape
    krows, kcols = kernel.shape

    rows = rows - krows + 1
    cols = cols - kcols + 1

    conv_image = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            conv_image[i, j] = np.sum(img[i:i+krows, j:j+kcols]*kernel)

    return conv_image