from matplotlib import cm
import matplotlib as mpl
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


def displayData(X, example_width=np.array([])):
    # DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    # Set example_width automatically if not passed in
    if 'example_width' not in locals() or np.size(example_width) == 0:
        example_width = (np.round(np.sqrt(np.size(X, 1)))).astype('int32')

    # Gray Image
    color = mpl.colormaps['gist_gray']

    # Compute rows, cols
    m, n = np.shape(X)
    example_height = (n / example_width).astype('int32')

    # Compute number of items to display
    display_rows = (np.floor(np.sqrt(m))).astype('int32')
    display_cols = (np.ceil(m / display_rows)).astype('int32')

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    currEx = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if currEx > m:
                break

            # Copy the patch

            # Get the max value of the patch
            max_val = np.max(np.abs(X[currEx, :]))
            startj, endj = pad + j * (example_height + pad) + (0, example_width)
            starti, endi = pad + i * (example_height + pad) + (0, example_height)
            display_array[startj:endj, starti:endi] = \
                np.reshape(X[currEx, :], (example_height, example_width), order='F') / max_val
            currEx = currEx + 1

        if currEx > m:
            break

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1, cmap=color)

    # Do not show axis
    plt.axis('off')
    plt.show()

    return h, display_array
