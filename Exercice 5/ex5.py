# Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization

import numpy as np
from scipy.io import loadmat
import functions as f
import matplotlib.pyplot as plt

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = loadmat('../Data/ex5data1.mat')
X = data['X']
Xtest = data['Xtest']
Xval = data['Xval']
y = data['y'].reshape((np.size(data['y']),))
ytest = data['ytest'].reshape((np.size(data['ytest']),))
yval = data['yval'].reshape((np.size(data['yval']),))

# m = Number of examples
m = np.size(X, 0)

# Plot training data
plt.plot(X, y, 'rx', ms=10, lw=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#
Xc = np.concatenate((np.ones((m, 1)), X), axis=1)
theta = np.array([1, 1])
J = f.linearRegCostFunction(Xc, y, theta, 1)

print('Cost at theta = [1  1]: {:2f}'
    '\n(this value should be about 303.993192)\n'.format(J))

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([1, 1])
J = f.linearRegCostFunction(Xc, y, theta, 1)
grad = f.linearRegGradCostFunction(Xc, y, theta, 1)

print('Gradient at theta = [1  1]:  [{:2f} {:2f}] '
    '\n(this value should be about [-15.303016 598.250744])\n'.format(grad[0], grad[1]))

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
lambda_t = 0
theta, cost = f.trainLinearReg(Xc, y, lambda_t)

#  Plot fit over the data
plt.plot(X, y, 'rx', ms=10, lw=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, Xc @ theta, lw=2)
plt.show()

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

lambda_t = 0
Xvalc = np.concatenate((np.ones((np.size(Xval, 0), 1)), Xval), axis=1)
error_train, error_val = f.learningCurve(Xc, y, Xvalc, yval, lambda_t)

plt.plot(np.arange(1, m+1), error_train, label='Train')
plt.plot(np.arange(1, m+1), error_val, label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim([0, 13])
plt.ylim([0, 150])
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
# for i in range(m):
#     print('\t{:d}\t\t{:2f}\t{:2f}\n'.format(i, error_train[i], error_val[i]))

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = f.polyFeatures(X, p)
X_poly, mu, sigma = f.featureNormalize(X_poly)  # Normalize
X_poly = np.concatenate((np.ones((m, 1)), X_poly), axis=1)                  # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = f.polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate((np.ones((np.size(X_poly_test, 0), 1)), X_poly_test), axis=1)      # Add Ones


# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = f.polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate((np.ones((np.size(X_poly_val, 0), 1)), X_poly_val), axis=1)         # Add Ones

print('Normalized Training Example 1:\n')
# print('  {:2f}  \n'.format(X_poly[0, :]))

print('\nProgram paused. Press enter to continue.\n')
input('Press enter to continue')


# =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda_t = 0
theta, cost = f.trainLinearReg(X_poly, y, lambda_t)

# Plot training data and fit
# figure(1)
plt.plot(X, y, 'rx', ms=10, lw=1.5)
f.plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {:2f})'.format(lambda_t))
plt.show()

error_train, error_val = f.learningCurve(X_poly, y, X_poly_val, yval, lambda_t)
plt.plot(np.arange(1, m + 1), error_train, label='Train')
plt.plot(np.arange(1, m + 1), error_val, label='Cross Validation')

plt.title('Polynomial Regression Learning Curve (lambda = {:2f})'.format(lambda_t))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim([0, 13])
plt.ylim([0, 100])
plt.legend()
plt.show()

print('Polynomial Regression (lambda = {:2f})\n\n'.format(lambda_t))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
# for i in range(m):
#     print('\t{:d}\t\t{:2f}\t{:2f}\n'.format(i, error_train[i], error_val[i]))

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = f.validationCurve(X_poly, y, X_poly_val, yval)


plt.plot(lambda_vec, error_train, label='Train')
plt.plot(lambda_vec, error_val, label='Cross Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('lambda\t\tTrain Error\tValidation Error\n')

# for i in range(len(lambda_vec)):
#     print(' {:2f}\t{:2f}\t{:2f}\n'.format(lambda_vec[i], error_train[i], error_val[i]))


print('Program paused. Press enter to continue.\n')
input('Press enter to continue')
