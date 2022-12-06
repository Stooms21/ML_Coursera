# Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization
import numpy as np
import spamUtils as sU
from scipy.io import loadmat
from sklearn import svm


# ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('\nPreprocessing sample email (emailSample1.txt)\n')

# Extract Features
emailSample1 = open('../Data/emailSample1.txt', 'r')
file_contents = emailSample1.read()
emailSample1.close()
word_indices = sU.processEmail(file_contents)

# Print Stats
print('Word Indices: \n')
print(f' {word_indices}')
print('\n\n')

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
features = sU.emailFeatures(word_indices)

# Print Stats
print(f'Length of feature vector: {len(features)}\n')
print(f'Number of non-zero entries: {sum(features > 0)}\n')

print('Program paused. Press enter to continue.\n')
input('Press enter to continue')

# =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat('../Data/spamTrain.mat')
X = data['X']
y = data['y'].flatten()


print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
model = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=-1)
model.fit(X, y)

p = model.predict(X)

print(f'Training Accuracy: {np.mean(p == y)*100}\n')

# =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
data = loadmat('../Data/spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

print('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(Xtest)

print(f'Test Accuracy: {np.mean(p == ytest)*100} \n')
input('Press enter to continue')


# ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
idx = (-model.coef_[0]).argsort()[:len(model.coef_[0])]
weight = np.sort(model.coef_[0])[::-1]
vocabList = sU.getVocabList()

print('\nTop predictors of spam: \n')
for i in range(15):
    print(f' {vocabList[idx[i]]:10} {weight[i]} \n')

print('\n\n')
print('\nProgram paused. Press enter to continue.\n')
input('Press enter to continue')

# =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
# filename = 'spamSample1.txt'
#
# # Read and predict
# file_contents = readFile(filename)
# word_indices  = processEmail(file_contents)
# x             = emailFeatures(word_indices)
# p = svmPredict(model, x)
#
# print('\nProcessed #s\n\nSpam Classification: #d\n', filename, p)
# print('(1 indicates spam, 0 indicates not spam)\n\n')

