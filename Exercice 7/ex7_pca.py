# Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
from scipy.io import loadmat
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.pyplot as plt
import pcafuncs as pcf
import plotfuncs as plf
import numpy as np
import functions as f

# Initialization

# ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.\n\n')

#  The following command loads the dataset. You should now have the
#  variable X in your environment
data = loadmat('../Data/ex7data1.mat')
X = data['X']

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('\nRunning PCA on example dataset.\n\n')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = pcf.featureNormalize(X)

#  Run PCA
U, S = pcf.pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
plf.drawLine(mu, mu + 1.5 * S[0] * U[:, 0].T)
plf.drawLine(mu, mu + 1.5 * S[1] * U[:, 1].T)
plt.show()

print('Top eigenvector: \n')
print(f' U(:,1) = {U[0, 0]} {U[1, 0]} \n')
print('\n(you should expect to see -0.707107 -0.707107)\n')

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#  first k eigenvectors. The code will then plot the data in this reduced
#  dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('\nDimension reduction on example dataset.\n\n')

#  Plot the normalized dataset (returned from pca)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4, 3, -4, 3])
plt.show()

#  Project the data onto K = 1 dimension
K = 1
Z = pcf.projectData(X_norm, U, K)
print('Projection of the first example: \n', Z[0])
print('\n(this value should be about 1.481274)\n\n')

X_rec = pcf.recoverData(Z, U, K)
print(f'Approximation of the first example: {X_rec[0, 0]} {X_rec[0, 1]}\n')
print('\n(this value should be about  -1.047419 -1.047419)\n\n')

#  Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(np.size(X_norm, 0)):
    plf.drawLine(X_norm[i, :], X_rec[i, :])

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('\nLoading face dataset.\n\n')

#  Load Face dataset
data = loadmat('../Data/ex7faces.mat')
X = data['X']

#  Display the first 100 faces in the dataset
plf.displayData(X[0:100, :])

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('\nRunning PCA on face dataset.\n'
      '(this might take a minute or two ...)\n\n')

#  Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
X_norm, mu, sigma = pcf.featureNormalize(X)

#  Run PCA
U, S = pcf.pca(X_norm)

#  Visualize the top 36 eigenvectors found
plf.displayData(U[:, 0:36].T)

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
print('\nDimension reduction for face dataset.\n\n')

K = 100
Z = pcf.projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(np.size(Z))

print('\n\nProgram paused. Press enter to continue.\n')
input('Press enter to continue')

# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 100
X_rec = pcf.recoverData(Z, U, K)

# Display normalized data
ax1 = plt.subplot(1, 2, 1)
plf.displayData(X_norm[0:100, :])
plt.title('Original faces')
ax1.remove()

# Display reconstructed data from only k eigenfaces
ax2 = plt.subplot(1, 2, 2)
plf.displayData(X_rec[0:100, :])
plt.title('Recovered faces')
ax2.remove()
plt.show()

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.


# Reload the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = mpimg.imread('../Data/bird_small.png', format='png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat');

img_size = A.shape
X = np.reshape(A, (img_size[0] * img_size[1], 3), order='F')
K = 16
max_iters = 10
initial_centroids = f.kMeansInitCentroids(X, K)
[centroids, idx] = f.runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = (np.floor(np.random.rand(1000, 1) * np.size(X, 0)) + 1).astype('int32')

#  Setup Color Palette
hsv = cm.get_cmap('hsv', K)
hsv = hsv(range(K))
colors = hsv[idx[sel], :]

#  Visualize the data and centroid memberships in 3D
plt.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()
print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = pcf.featureNormalize(X)

# PCA and project the data to 2D
U, S = pcf.pca(X_norm)
Z = pcf.projectData(X_norm, U, 2)

# Plot in 2D
plf.plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
print('Program paused. Press enter to continue.\n')
input('Press enter to continue')
