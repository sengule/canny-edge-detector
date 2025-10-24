import numpy as np

#kernel source : https://en.wikipedia.org/wiki/Kernel_(image_processing)
class Kernels:
    #3x3 kernels
    kernels = {
        "identity": np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=float),

        "ridge": np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=float),

        "edge": np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=float),

        "sharpen": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=float),

        "box_blur": np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=float) / 9.0,

        "gaussian_blur": np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=float) / 16.0,

        "sobel":(
            np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ], dtype=float),

            np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ], dtype=float)
        ),

        "prewitt":(
            np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ], dtype=float),

            np.array([
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1],
            ], dtype=float)
        )

    }

    @classmethod
    def get_kernel(self, type):
        kernel_type = type.lower()
        return self.kernels[kernel_type]

    def gaussian_kernel(l=5, sig=1.):
        """
        creates gaussian kernel with creating 1xl gaussian array and
        it takes outer product of itself. it results lxl gaussian kernel.
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)

        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))

        kernel = np.outer(gauss, gauss)

        #normalized result
        return kernel / np.sum(kernel)