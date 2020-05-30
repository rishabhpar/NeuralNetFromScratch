import numpy as np
from sigmoids import sigmoid

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    first = np.concatenate([np.ones((m, 1)), X], axis=1)
    second = Theta1.T
    h1 = sigmoid(np.dot(first, second))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p
