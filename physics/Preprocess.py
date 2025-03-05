import numpy as np

class PolyLinealizer:
    def __init__(self, degree):
        self.degree = degree

    def fit_transform(self, x):
        n = len(x)
        x_l = np.column_stack([x**i for i in range (0, self.degree + 1)])
        return x_l