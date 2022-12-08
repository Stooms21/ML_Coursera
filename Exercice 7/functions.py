import numpy as np
import plotfuncs as pf
import matplotlib.pyplot as plt


def findClosestCentroids(X, centroids):
    # FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    #   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    #   in idx for a dataset X where each row is a single example. idx = m x 1 
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #
    K: int = np.size(centroids, 0)
    idx = np.zeros(np.size(X, 0))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #
    for i in range(np.size(X, 0)):
        idx[i] = np.argmin(np.linalg.norm(X[i, :] - centroids, axis=1)**2)
    # =============================================================

    return idx.astype('int32')


def computeCentroids(X, idx, K: int):
    # COMPUTECENTROIDS returns the new centroids by computing the means of the
    # data points assigned to each centroid.
    #   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
    #   computing the means of the data points assigned to each centroid. It is
    #   given a dataset X where each row is a single data point, a vector
    #   idx of centroid assignments (i.e. each entry in range [1..K]) for each
    #   example, and K, the number of centroids. You should return a matrix
    #   centroids, where each row of centroids is the mean of the data points
    #   assigned to it.
    #

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids(i, :)
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # Note: You can use a for-loop over the centroids to compute this.
    #
    for i in range(K):
        centroids[i, :] = np.mean(X[idx == i, :], axis=0)
    # =============================================================
    return centroids


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    # is a single example
    #   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    #   plot_progress) runs the K-Means algorithm on data matrix X, where each
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions
    #   of K-Means to execute. plot_progress is a true/false flag that
    #   indicates if the function should also plot its progress as the
    #   learning happens. This is set to false by default. runkMeans returns
    #   centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set default value for plot progress

    # Plot the data if we are plotting progress
    # if plot_progress:
    # #     figure;
    # #     hold on;
    # # end

    # Initialize values
    m, n = X.shape
    K: int = np.size(initial_centroids, 0)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print(f'K-Means iteration {i+1}/{max_iters}...\n')

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            pf.plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            print('Press enter to continue.\n')
            input('Press enter to continue')

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    return centroids, idx


def kMeansInitCentroids(X, K: int):
    # KMEANSINITCENTROIDS This function initializes K centroids that are to be
    # used in K-Means on the dataset X
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X
    #
    
    # You should return this values correctly
    centroids = np.zeros((K, np.size(X, 1)))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #
    
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randIdx = np.random.permutation(np.size(X, 0))

    # Take the first K examples as centroids
    centroids = X[randIdx[:K], :]
    # =============================================================
    return centroids
