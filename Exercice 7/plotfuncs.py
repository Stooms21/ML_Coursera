from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


def plotDataPoints(X, idx, K):
    # PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    #   with the same index assignments in idx have the same color

    # Create palette
    hsv = cm.get_cmap('hsv', K + 1)
    hsv = hsv(range(K + 1))
    # palette = hsv(K + 1)
    colors = hsv[idx, :]

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], s=15, c=colors)


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # PLOTPROGRESSKMEANS is a helper function that displays the progress of
    # k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    #

    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'x', mec='k', ms=10, lw=3)

    # Plot the history of the centroids with lines
    for j in range(np.size(centroids, 0)):
        drawLine(centroids[j, :], previous[j, :])

    # Title
    plt.title(print('Iteration number ', i))
    plt.show()


def drawLine(p1, p2, *args):
    # DRAWLINE Draws a line from point p1 to point p2
    #   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
    #   current figure

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], args[:])
