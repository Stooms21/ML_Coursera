import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_t):

    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network

    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)), order='F')

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1))::],
                        (num_labels, (hidden_layer_size + 1)), order='F')

    # Setup some useful variables
    m = np.size(X, 0)

    # recode the y labels to get vectors of 0 and 1
    eye = np.eye(num_labels)
    yr = eye[y - 1]

    # Add the column of ones for bias unit
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Compute first hidden unit and adding the bias
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(X @ Theta1.T)), axis=1)

    # Compute h_theta
    h_theta = sigmoid(a2 @ Theta2.T)

    # You need to return the following variables correctly
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recomm implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #




    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradients
    J = 1 / m * np.sum(-yr * np.log(h_theta) - (1 - yr) * np.log(1 - h_theta))

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

