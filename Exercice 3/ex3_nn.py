# Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization
import numpy as np
from scipy.io import loadmat
import functions as f

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')
data = loadmat('../Data/ex3data1.mat')
X = data['X']
y = data['y'].reshape((np.size(data['y']),))
m = np.size(X, 0)

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[0:100]

f.displayData(X[sel, :])

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weights = loadmat('../Data/ex3weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

# ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = f.predict(theta1, theta2, X)

print('\nTraining Set Accuracy: #f\n', np.mean(pred == y) * 100)

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in range(m):
    # Display
    print('\nDisplaying Example Image\n')
    number = (X[rp[i], :].reshape((np.size(X, 1), 1))).T
    f.displayData(number)

    pred = f.predict(theta1, theta2, number)
    print('\nNeural Network Prediction: #d (digit #d)\n', pred, pred % 10)

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
