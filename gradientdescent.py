from cost import nnCostFunction
import numpy as np


def gradientDescentnn(X, y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size, hidden_layer_size, num_labels):
    Theta1 = initial_nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size,
                                                                                      input_layer_size + 1)
    Theta2 = initial_nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

    V_dT1 = np.zeros(Theta1.shape)
    V_dT2 = np.zeros(Theta2.shape)
    beta = .9

    for i in range(num_iters):
        nn_params = np.append(Theta1.flatten(), Theta2.flatten())
        cost, grad1, grad2 = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

        V_dT1 = (beta * V_dT1) + (1. - beta) * grad1
        V_dT2 = (beta * V_dT2) + (1. - beta) * grad2

        Theta1 = Theta1 - (alpha * V_dT1)
        Theta2 = Theta2 - (alpha * V_dT2)

        if i % 100 == 0:
            print("Epoch {}: test cost = {}".format(i, cost))

    return Theta1, Theta2
