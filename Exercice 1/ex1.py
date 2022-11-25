import matplotlib.pyplot as plt
import functions as f
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# ==================== Part 1: Basic Function ====================

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
f.warmUpExercise()

print('Program paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.transpose(np.loadtxt('../Data/ex1data1.txt', delimiter=","))
X = data[0]
y = data[1]
m = len(y)  # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
f.plotData(X, y)

print('Program paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# =================== Part 3: Cost and Gradient descent ===================

X = np.column_stack((np.ones(m), X))  # Add a column of ones to x
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = f.computeCost(X, y, theta)
print('With theta = [0  0]\nCost computed = #f\n', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = f.computeCost(X, y, np.array([-1, 2]))
print('\nWith theta = [-1  2]\nCost computed = #f\n', J)
print('Expected cost value (approx) 54.24\n')

print('Program paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, J_history = f.gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('#f\n', theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(X[:, 1], theta.dot(np.transpose(X)), '-', label="Linear regression")
plt.plot(X[:, 1], y, "o", label="Training data")
plt.xlim(left=0)
plt.xlabel("Population")
plt.ylabel("Revenue")
plt.legend(loc="upper left")
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of #f\n',
	predict1 * 10000)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of #f\n',
	predict2 * 10000)

print('Program paused. Press enter to continue.\n')
wait = input("Press Enter to continue.")

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
	for j in range(len(theta1_vals)):
		t = np.array([theta0_vals[i], theta1_vals[j]])
		J_vals[i][j] = f.computeCost(X, y, t)


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals)
# Surface plot

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1,
	cmap=cm.coolwarm, antialiased=True)
ax.set_xlabel("theta0")
ax.set_ylabel("theta1")
ax.set_zlabel("J")
plt.show()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
fig, ax = plt.subplots(1, 1)
cp = ax.contour(theta0_vals, theta1_vals, J_vals,
	levels=np.logspace(-2, 3, 20), cmap=cm.coolwarm)
ax.set_xlabel("theta0")
ax.set_ylabel("theta1")
fig.colorbar(cp)
plt.plot(theta[0], theta[1], 'x', markersize=12, color='red')
plt.show()
