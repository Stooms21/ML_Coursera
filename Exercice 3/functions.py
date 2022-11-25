import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy.optimize import fmin_cg


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


def lrCostFunction(theta, X, y, lambda_t):
    # LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    # regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

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
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #
    # Hint: When computing the gradient of the regularized cost function,
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta;
    #           temp(1) = 0;   # because we don't add anything for j = 0
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)
    #
    h_theta = sigmoid(X.dot(theta))
    J = 1. / m * sum(-y * np.log(h_theta) - (1. - y) * np.log(1. - h_theta)) \
        + lambda_t / (2. * m) * sum(theta[1::] ** 2)
    return J


def lrGradCostFunction(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros(np.shape(theta))

    h_theta = sigmoid(X.dot(theta))
    grad[0] = 1. / m * (h_theta - y).dot(X[:, 0])
    grad[1::] = 1. / m * (h_theta - y).dot(X[:, 1::]) + lambda_t / m * theta[1::]
    return grad


def oneVsAll(X, y, num_labels, reg):
    # ONEVSALL trains multiple logistic regression classifiers and returns all
    # the classifiers in a matrix all_theta, where the i-th row of all_theta
    # corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logistic regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     # Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #
    #     # Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    #
    #     # Run fmincg to obtain the optimal theta
    #     # This function will return theta and the cost
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);
    #
    for i in range(1, num_labels+1):
        temp = np.where(y == i, 1, 0)
        output = fmin_cg(lrCostFunction, all_theta[i-1, :], lrGradCostFunction,
                      args=(X, temp, reg),
                      maxiter=100,
                      full_output=True)

        all_theta[i-1, :] = output[0]
        del temp

    return all_theta


def predictOneVsAll(all_theta, X):
    # Predict the label for a trained one-vs-all classifier. The labels
    # are in the range 1..K, where K = size(all_theta, 1).
    #  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #  for each example in the matrix X. Note that X contains the examples in
    #  rows. all_theta is a matrix where the i-th row is a trained logistic
    #  regression theta vector for the i-th class. You should set p to a vector
    #  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    #  for 4 examples)

    m = np.size(X, 0)
    num_labels = np.size(all_theta, 0)

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    p = sigmoid(X.dot(all_theta.T))
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set p to a vector of predictions (from 1 to
    #               num_labels).
    #
    # Hint: This code can be done all vectorized using the max function.
    #       In particular, the max function can also return the index of the
    #       max element, for more information see 'help max'. If your examples
    #       are in rows, then, you can use max(A, [], 2) to obtain the max
    #       for each row.
    #

    # =========================================================================
    return np.argmax(p, axis=1) + 1


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def predict(theta1, theta2, X):
    #   PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = np.size(X, 0)
    num_labels = np.size(theta2, 0)

    # You need to return the following variables correctly
    p = np.zeros((np.size(X, 0), 1))

    # Set bias unit
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    #

    # =========================================================================

    a2 = sigmoid(X @ theta1.T)
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)

    a3 = sigmoid(a2 @ theta2.T)
    p = np.argmax(a3, axis=1) + 1

    return p
