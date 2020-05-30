import numpy as np
from sigmoids import sigmoidGradient, sigmoid

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X_train, Y_train, lambdaVal):
    m = X_train.shape[0]

    theta_1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    theta_2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    a1 = np.c_[np.ones(m),X_train]
    z2 = np.dot(theta_1, a1.T)
    a2 = np.c_[np.ones(m), sigmoid(z2).T]
    z3 = np.dot(theta_2, a2.T)
    hypothesis = sigmoid(z3)

    firsthalf = -1 * np.multiply(Y_train, np.log(hypothesis))
    secondhalf = np.multiply((1 - Y_train), np.log(1 - hypothesis))

    cost = np.sum(firsthalf - secondhalf) / m +\
        (lambdaVal / (2 * m)) * (np.sum(np.square(theta_1[:, 1:])) +
            np.sum(np.square(theta_2[:, 1:])))

    d3 = (hypothesis - Y_train).T
    d2 = theta_2[:,1:].T.dot(d3.T) * sigmoidGradient(z2)

    delta1 = np.dot(d2, a1)
    delta2 = np.dot(d3.T, a2)

    theta1_ = np.c_[np.ones((theta_1.shape[0], 1)), theta_1[:, 1:]]
    theta2_ = np.c_[np.ones((theta_2.shape[0], 1)), theta_2[:, 1:]]

    theta1_grad = delta1 / m + (theta1_ * lambdaVal) / m
    theta2_grad = delta2 / m + (theta2_ * lambdaVal) / m

    grad = np.concatenate([theta1_grad.ravel(), theta2_grad.ravel()])

    # return cost, theta1_grad, theta2_grad
    return cost, grad
