import numpy as np


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

