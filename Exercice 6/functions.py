import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y, plot=True):
    # PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples

    # Plot Examples
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'k+', lw=1, ms=7)
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'ko', mfc='y', ms=7)
    if plot:
        plt.show()


def visualizeBoundaryLinear(X, y, model):
    # VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the SVM
    # VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    # learned by the SVM and overlays the data on it

    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = - (w[0] * xp + b) / w[1]
    plotData(X, y, False)
    plt.plot(xp, yp, '-b')
    plt.show()


def gaussianKernel(x1, x2, sigma):
    # RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = x1[:]
    x2 = x2[:]

    # You need to return the following variables correctly.
    sim = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #
    sim = np.exp(-(np.linalg.norm(x1 - x2))**2/(2 * sigma**2))
    return sim
    # =============================================================


def visualizeBoundary(X, y, model):
    # VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    # VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    # boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    plotData(X, y, False)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(np.size(X1, 1)):
        this_X = np.array([X1[:, i], X2[:, i]])
        vals[:, i] = model.predict(this_X.T)

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, levels=[0.5, 0.50000001], colors='b')
    plt.show()


