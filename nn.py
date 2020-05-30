import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report
from predict import predict
from gradientdescent import gradientDescentnn

### IMPORTANT CONSTANTS ###
input_layer_size = 784  # 28x28 Input Images of Digits
hidden_layer_size = 64  # 25 hidden units
num_labels = 10         # 10 classes, from 1 to 10
lambdaVal = 2           # regularization constant


### LOADING DATA ###
mnist = loadmat('mnist-original.mat')
X = mnist["data"]
y = mnist["label"].ravel()
y[y == 10] = 0         # set the zero digit to 0, rather than its mapped 10 in this dataset
                       # This is an artifact due to the fact that this dataset was used in
                       # MATLAB where there is no index 0

### NORMALIZE FEATURES ###
X = (X / 255).T

### ONE-HOT ENCODE Y ###
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)
Y_new = Y_new.T


### CREATE A TRAIN, VALIDATION, AND TEST SET ###
m = 60000
m_test = 10000
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:m], Y_new[m:]
X_train = X_train.T
X_test = X_test.T
Y_test = Y_test.T
Y_train = Y_train.T

### RANDOMLY INITIALIZE WEIGHTS ###
theta_1 = np.random.randn(hidden_layer_size, 1 + input_layer_size) * np.sqrt(1/input_layer_size)
theta_1[:, 0] = 0
theta_2 = np.random.randn(num_labels, 1 + hidden_layer_size) * np.sqrt(1/hidden_layer_size)
theta_2[:, 0] = 0
random_nn_params = np.concatenate([theta_1.ravel(), theta_2.ravel()], axis=0)


### RUN OPTIMIZATION ON WEIGHTS AND BIASES ###
Theta1, Theta2 = gradientDescentnn(X_train, Y_train, random_nn_params, 16, 2000, lambdaVal, input_layer_size,
                                   hidden_layer_size, num_labels)

### DETERMINE ACCURACY ###
print(classification_report(predict(Theta1, Theta2, X_test), np.argmax(Y_test, axis=0)))
