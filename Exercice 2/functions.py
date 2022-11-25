import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Create New Figure

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #
    Xpos = X[:][y == 1]
    Xneg = X[:][y == 0]
    plt.plot(Xpos[:, 0], Xpos[:, 1], '+', color='black',
             label='Admitted')
    plt.plot(Xneg[:, 0], Xneg[:, 1], 'o', color='orange',
             label='Not admitted')


# =========================================================================


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.

    # You need to return the following variables correctly
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).
    return 1. / (1. + np.exp(-z))


# =============================================================


def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    # J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    J = 1. / m * sum(-y * np.log(sigmoid(theta.dot(np.transpose(X))))
                     - (1. - y) * np.log(1. - sigmoid(theta.dot(np.transpose(X)))))
    return J


# =============================================================


def gradCostFunction(theta, X, y):
    m = len(y)  # number of training examples
    grad = np.zeros(np.size(theta))
    grad = 1. / m * (sigmoid(theta.dot(np.transpose(X))) - y).dot(X)
    return grad


def plotDecisionBoundary(theta, X, y):
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    plotData(X[:, 1:3], y)

    if len(X[1, :]) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, label='Decision Boundary')

        # Legend, specific for the exercise
        plt.legend()
        plt.xlim(30, 100)
        plt.ylim(30, 100)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeaturePlot(u[i], v[j], 6).dot(theta)
        z = np.transpose(z)  # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        # fig, ax = plt.subplots(1, 1)
        plt.contour(u, v, z, 0,
                cmap=cm.coolwarm)


def mapFeature(X1, X2, degree):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #
    out = np.ones(np.size(X1)).reshape(np.size(X1), 1)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            terms = (X1 ** (i - j) * X2 ** j).reshape(np.size(X1), 1)
            out = np.hstack((out, terms))
    return out


def mapFeaturePlot(x1, x2, degree):
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            terms = (x1**(i-j) * x2**j)
            out = np.hstack((out, terms))
    return out


def predict(theta, X):
    m = len(X[:, 1])
    p = np.zeros(m)
    p = sigmoid(theta.dot(np.transpose(X)))
    p[p < 0.5] = 0.
    p[p >= 0.5] = 1.
    return p


def costFunctionReg(theta, X, y, regParam):
    m = len(y)
    J = 1. / m * sum(-y * np.log(sigmoid(theta.dot(np.transpose(X))))
        - (1. - y) * np.log(1. - sigmoid(theta.dot(np.transpose(X))))) \
        + regParam / (2. * m) * sum(theta[1::] ** 2)
    return J


def gradCostFunctionReg(theta, X, y, regParam):
    m = len(y)  # number of training examples
    grad = np.zeros(np.size(theta))
    grad[0] = 1. / m * (sigmoid(theta.dot(np.transpose(X))) - y).dot(X[:, 0])
    grad[1::] = 1. / m * (sigmoid(theta.dot(np.transpose(X))) - y)\
        .dot(X[:, 1::]) + regParam / m * theta[1::]
    return grad
