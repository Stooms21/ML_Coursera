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
import mping as mping
# Initialization
import numpy as np
import functions as f
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you should complete the code in the findClosestCentroids function.
#
print('Finding closest centroids.\n\n')

# Load an example dataset that we will be using
data = loadmat('../Data/ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K: int = 3  # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = f.findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(f'{idx[:3]}')
print('\n(the closest centroids should be 1, 3, 2 respectively)\n')

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('\nComputing centroids means.\n\n')

#  Compute means based on the closest centroids found in the previous part.
centroids = f.computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(f' {centroids} \n')
print('\n(the centroids should be\n')
print('   [ 2.428301 3.157924 ]\n')
print('   [ 5.813503 2.633656 ]\n')
print('   [ 7.119387 3.616684 ]\n\n')

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')


# =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided.
#
print('\nRunning K-Means clustering on example dataset.\n\n')

# Load an example dataset
data = loadmat('../Data/ex7data2.mat')
X = data['X']

# Settings for running K-Means
K: int = 3
max_iters: int = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = f.runkMeans(X, initial_centroids, max_iters, False)
print('\nK-Means Done.\n\n')

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel onto its closest centroid.
#
#  You should now complete the code in kMeansInitCentroids.m
#

print('\nRunning K-Means clustering on pixels from an image.\n\n')

#  Load an image of a bird
A = mpimg.imread('../Data/touf.png', format="png")

# If imread does not work for you, you can try instead
#   load ('bird_small.mat');


# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = np.reshape(A, (img_size[0] * img_size[1], 3), order='F')

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
# Display the original image
plt.subplot(2, 3, (1, 1))
plt.imshow(A)
plt.title('Original')

for i in range(1, 6):
    K: int = 2**i
    max_iters: int = 10

    # When using K-Means, it is important the initialize the centroids
    # randomly.
    # You should complete the code in kMeansInitCentroids.m before proceeding
    initial_centroids = f.kMeansInitCentroids(X, K)

    # Run K-Means
    centroids, idx = f.runkMeans(X, initial_centroids, max_iters)

    print('Program paused. Press enter to continue.\n')
    input('Press enter to continue')

    # ================= Part 5: Image Compression ======================
    #  In this part of the exercise, you will use the clusters of K-Means to
    #  compress an image. To do this, we first find the closest clusters for
    #  each example. After that, we

    print('\nApplying K-Means to compress an image.\n\n')

    # Find closest cluster members
    idx = f.findClosestCentroids(X, centroids)

    # Essentially, now we have represented the image X as in terms of the
    # indices in idx.

    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by its index in idx) to the centroid value
    X_recovered = centroids[idx, :]

    # Reshape the recovered image into proper dimensions
    X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3), order='F')

    # Display compressed image side by side
    plt.subplot(2, 3, (i // 3 + 1, i % 3 + 1))
    plt.imshow(X_recovered)
    plt.title(f'Compressed, with {K} colors.')
plt.show()


print('Program paused. Press enter to continue.\n')
input('Press enter to continue')
