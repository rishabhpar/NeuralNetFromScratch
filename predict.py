import numpy as np
from sigmoids import sigmoid

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    first = np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T)
    h1 = sigmoid(first)
    second = np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T)
    h2 = np.exp(second) / np.sum(np.exp(second), axis=0)
    p = np.argmax(h2, axis=1)
    return p
