import numpy as np
from kernel import Kernels
from convolution import convolve2d

def _gaussian_blur(img):
    kernel = Kernels.gaussian_kernel(l=5, sig=1.5)
    return convolve2d(img, kernel)

def _sobel_operator(img):
    Gx, Gy = Kernels.get_kernel('sobel')

    gx=convolve2d(img, Gx)
    gy=convolve2d(img, Gy)

    return np.sqrt(gx ** 2 + gy ** 2),gx,gy

def _nms(gradient_magnitude, gradient_direction):

    rows, cols = gradient_magnitude.shape
    nms_image = np.zeros((rows, cols), dtype=np.float32)

    angle = np.rad2deg(gradient_direction) % 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            #0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]

            #45
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]

            #90
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]

            #135
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]


            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                nms_image[i, j] = gradient_magnitude[i, j]
            else:
                nms_image[i, j] = 0

    return nms_image

def _double_threshold(img,low_th, high_th):

    strong = np.uint8(255)
    weak = np.uint8(0)

    res = np.zeros_like(img, dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high_th)
    weak_i, weak_j = np.where((img >= low_th) & (img < high_th))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def _hysteresis(img, weak=50, strong=100):

    rows, cols = img.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img[i, j] == weak:
                # Check 8-connected neighbors
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or
                        (img[i + 1, j + 1] == strong) or (img[i, j - 1] == strong) or
                        (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong) or
                        (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img

def canny_edge_detector(img, low_th = 5, high_th= 60):

    """
    Steps:
    1. Gaussian blur
    2. Sobel operator
    3. Non-maximum suppression
    4. Double thresholding
    5. Hysteresis
    """
    gaussian_blur = _gaussian_blur(img)

    grad_mag,gx,gy = _sobel_operator(gaussian_blur)
    theta = np.arctan2(gy, gx)

    nms = _nms(grad_mag, theta)

    double_threshold = _double_threshold(nms,low_th, high_th)

    hys = _hysteresis(double_threshold)

    return hys.astype(np.uint8)


