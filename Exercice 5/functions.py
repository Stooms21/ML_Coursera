import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg


def featureNormalize(X):
    # FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    
    mu = np.mean(X)
    X_norm = X - mu
    
    sigma = np.std(X_norm)
    X_norm /= sigma

    return X_norm, mu, sigma


def learningCurve(X, y, Xval, yval, lambda_t):
    # LEARNINGCURVE Generates the train and cross validation set errors needed
    # to plot a learning curve
    #   [error_train, error_val] = ...
    #       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
    #       cross validation set errors for a learning curve. In particular,
    #       it returns two vectors of the same length - error_train and
    #       error_val. Then, error_train(i) contains the training error for
    #       i examples (and similarly for error_val(i)).
    #
    #   In this function, you will compute the train and test errors for
    #   dataset sizes from 1 up to m. In practice, when working with larger
    #   datasets, you might want to do this in larger intervals.
    #

# Number of training examples
    m = np.size(X, 0)

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in
    #               error_train and the cross validation errors in error_val.
    #               i.e., error_train(i) and
    #               error_val(i) should give you the errors
    #               obtained after training on i examples.
    #
    # Note: You should evaluate the training error on the first i training
    #       examples (i.e., X(1:i, :) and y(1:i)).
    #
    #       For the cross-validation error, you should instead evaluate on
    #       the _entire_ cross validation set (Xval and yval).
    #
    # Note: If you are using your cost function (linearRegCostFunction)
    #       to compute the training and cross validation error, you should
    #       call the function with the lambda argument set to 0.
    #       Do note that you will still need to use lambda when running
    #       the training to obtain the theta parameters.
    #
    # Hint: You can loop over the examples with the following:
    #
    #       for i = 1:m
    #           # Compute train/cross validation errors using training examples
    #           # X(1:i, :) and y(1:i), storing the result in
    #           # error_train(i) and error_val(i)
    #           ....
    #
    #       end
    #

    # ---------------------- Sample Solution ----------------------







    # -------------------------------------------------------------

    # =========================================================================

    return error_train, error_val


def linearRegCostFunction(X, y, theta, lambda_t):
    # LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    # regression with multiple variables
    #   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
    #   cost of using theta as the parameter for linear regression to fit the
    #   data points in X and y. Returns the cost in J and the gradient in grad

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly

    h_theta = X @ theta
    J = 1 / (2 * m) * sum((h_theta - y)**2) + lambda_t / (2 * m) * sum(theta[1::]**2)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
    # =========================================================================

    return J


def linearRegGradCostFunction(X, y, theta, lambda_t):

    m = len(y)
    grad = np.zeros(theta.shape)

    h_theta = X @ theta

    grad[0] = 1 / m * (h_theta - y).T @ X[:, 0]
    grad[1::] = 1 / m * (h_theta - y).T @ X[:, 1::] + lambda_t / m * theta[1::]

    return grad


def plotFit(min_x, max_x, mu, sigma, theta, p):
    # PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    # Also works with linear regression.
    # PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    # fit with power p and feature normalization (mu, sigma).

    # Hold on to the current figure

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = (min_x - np.arange(15, max_x + 25, 0.05).T)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate((np.ones((np.size(x, 0), 1)), X_poly), axis=1)

    # Plot
    plt.plot(x, X_poly @ theta, lw=2)


def polyFeatures(X, p):
    # POLYFEATURES Maps X (1D vector) into the p-th power
    #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

    # You need to return the following variables correctly.
    X_poly = np.zeros((np.size(X), p))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th
    #               column of X contains the values of X to the p-th power.
    #
    #
    # =========================================================================

    return X_poly


def trainLinearReg(X, y, lambda_t):
    # TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    # regularization parameter lambda
    #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta

    initial_theta = np.zeros((np.size(X, 1), 1))

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_t)
    gradCostFunction = lambda t: linearRegGradCostFunction(X, y, t, lambda_t)

    output = fmin_cg(costFunction, initial_theta, gradCostFunction,
            maxiter=200,
            full_output=True)
    theta = output[0]

    return theta


def validationCurve(X, y, Xval, yval):
    # VALIDATIONCURVE Generate the train and validation errors needed to
    # plot a validation curve that we can use to select lambda
    #   [lambda_vec, error_train, error_val] = ...
    #       VALIDATIONCURVE(X, y, Xval, yval) returns the train
    #       and validation errors (in error_train, error_val)
    #       for different values of lambda. You are given the training set (X,
    #       y) and validation set (Xval, yval).
    #

    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T

    # You need to return these variables correctly.
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in
    #               error_train and the validation errors in error_val. The
    #               vector lambda_vec contains the different lambda parameters
    #               to use for each calculation of the errors, i.e,
    #               error_train(i), and error_val(i) should give
    #               you the errors obtained after training with
    #               lambda = lambda_vec(i)
    #
    # Note: You can loop over lambda_vec with the following:
    #
    #       for i = 1:length(lambda_vec)
    #           lambda = lambda_vec(i);
    #           # Compute train / val errors when training linear
    #           # regression with regularization parameter lambda
    #           # You should store the result in error_train(i)
    #           # and error_val(i)
    #           ....
    #
    #       end
    #
    #
    # =========================================================================

    return lambda_vec, error_train, error_val
    
