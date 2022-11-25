# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization

import numpy as np
import functions as f
import matplotlib.pyplot as plt
from scipy.optimize import fmin_ncg

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).

data = np.transpose(np.loadtxt('../Data/ex2data2.txt', delimiter=","))
X = np.transpose(data[0:2])
y = data[2]

f.plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend()
plt.show()

# =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = f.mapFeature(X[:, 0], X[:, 1], 6)

# Initialize fitting parameters
initial_theta = np.zeros(len(X[1, :]))

# Set regularization parameter lambdaReg to 1
lambdaReg = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = f.costFunctionReg(initial_theta, X, y, lambdaReg)
grad = f.gradCostFunctionReg(initial_theta, X, y, lambdaReg)

np.set_printoptions(precision=3)

print('Cost at initial theta (zeros): #f\n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' #f \n', grad[0:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

print('\nProgram paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# Compute and display cost and gradient
# with all-ones theta and lambdaReg = 10
test_theta = np.ones(len(X[1, :]))
cost = f.costFunctionReg(test_theta, X, y, 10)
grad = f.gradCostFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambdaReg = 10): #f\n', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' #f \n', grad[0:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

print('\nProgram paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambdaReg and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambdaReg (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambdaReg? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros(len(X[1, :]))

# Set regularization parameter lambdaReg to 1 (you should vary this)
lambdaReg = 1

# Optimize
output = fmin_ncg(f.costFunctionReg, initial_theta, f.gradCostFunctionReg,
                  args=(X, y, 10),
                  maxiter=400,
                  full_output=True)

theta = output[0]
cost = output[1]

# Plot Boundary
f.plotDecisionBoundary(theta, X, y)
# title(sprintf('lambdaReg = #g', lambdaReg))

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend()
plt.show()

# Compute accuracy on our training set
p = f.predict(theta, X)

print('Train Accuracy: #f\n', np.mean((p == y)) * 100)
print('Expected accuracy (with lambdaReg = 1): 83.1 (approx)\n')
