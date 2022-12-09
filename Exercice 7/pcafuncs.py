import numpy as np


def featureNormalize(X):
    # FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis=0)
    X_norm /= sigma
    # ============================================================
    return X_norm, mu, sigma


def pca(X):
    # PCA Run principal component analysis on the dataset X
    #   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    #   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    #
    
    # Useful values
    m, n = X.shape
    
    # You need to return the following variables correctly.
    U = np.zeros((n, n))
    S = np.zeros((n, n))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix. 
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #
    
    sigma = X.T @ X / m
    U, S, V = np.linalg.svd(sigma)
    # =========================================================================
    return U, S


def projectData(X, U, K):
    # PROJECTDATA Computes the reduced data representation when projecting only
    # on to the top k eigenvectors
    #   Z = projectData(X, U, K) computes the projection of
    #   the normalized inputs X into the reduced dimensional space spanned by
    #   the first K columns of U. It returns the projected examples in Z.
    #

    # You need to return the following variables correctly.
    Z = np.zeros((np.size(X, 0), K))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the projection of the data using only the top K
    #               eigenvectors in U (first K columns).
    #               For the i-th example X(i,:), the projection on to the k-th
    #               eigenvector is given as follows:
    #                    x = X(i, :)';
    #                    projection_k = x' * U(:, k);
    #
    Z = X @ U[:, 0:K]
    # =============================================================
    return Z


def recoverData(Z, U, K):
    # RECOVERDATA Recovers an approximation of the original data when using the
    # projected data
    #   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
    #   original data that has been reduced to K dimensions. It returns the
    #   approximate reconstruction in X_rec.
    #

    # You need to return the following variables correctly.
    X_rec = np.zeros((np.size(Z, 0), np.size(U, 0)))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the approximation of the data by projecting back
    #               onto the original space using the top K eigenvectors in U.
    #
    #               For the i-th example Z(i,:), the (approximate)
    #               recovered data for dimension j is given as follows:
    #                    v = Z(i, :)';
    #                    recovered_j = v' * U(j, 1:K)';
    #
    #               Notice that U(j, 1:K) is a row vector.
    #
    X_rec = Z @ U[:, :K].T
    # =============================================================
    return X_rec
