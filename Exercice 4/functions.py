import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_t):
    J = 0
    return J


def nnGradCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_t):
    grad = 0
    return grad


def predict(Theta1, Theta2, X):
    p = 0
    return p


def randInitializeWeights(L_in, L_out):
    W = 0
    return W


def checkNNGradients(lambda_t=0):
    return 0


def computeNumericalGradient(J, theta):
    numgrad = 0
    return numgrad


def debugInitializeWeights(fan_out, fan_in):
    W = 0
    return W


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

    display_array = display_array

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1, cmap=color)

    # Do not show axis
    plt.axis('off')
    plt.show()

    return h, display_array


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoidGradient(z):
    g = 0
    return g

