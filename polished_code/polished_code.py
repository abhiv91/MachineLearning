from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report



#We will be using a fast gradient algorithm to solve an l2 regularized regress
#lets first import and standardize our data, the spam dataset

spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ',
                               header=None)

# the last column is the label as 0,and 1
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1  # Convert to +/- 1
test_indicator = np.array(test_indicator).T[0]

# Divide the data into train, test sets
x_train = x[test_indicator == 0, :]
x_test = x[test_indicator == 1, :]
y_train = y[test_indicator == 0]
y_test = y[test_indicator == 1]


# we can use sklearn to standardize our data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# number of samples and shape of each sample
n_train = len(y_train)
n_test = len(y_test)
d = np.size(x, 1)

#lets compute the gradient of the algorithm
def computegrad(beta, lambduh, x=x_train, y=y_train):
    yx = y[:, np.newaxis]*x
    denom = 1+np.exp(-yx.dot(beta))
    grad = 1/len(y)*np.sum(-yx*np.exp(-yx.dot(beta[:, np.newaxis]))/denom[:, np.newaxis], axis=0) + 2*lambduh*beta
    return grad


#Lets define the objective function
def objective(beta, lambduh, x=x_train, y=y_train):
    return 1/len(y) * np.sum(np.log(1 + np.exp(-y*x.dot(beta)))) + lambduh * np.linalg.norm(beta)**2


#lets use the backtracking ruele

def bt_line_search(beta, lambduh, eta=1, alpha=0.5, betaparam=0.8, maxiter=100, x=x_train, y=y_train):
    grad_beta = computegrad(beta, lambduh, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0
    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, x=x, y=y) < objective(beta, lambduh, x=x, y=y) \
                - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        elif iter == maxiter - 1:
            print('Warning: Max number of iterations of backtracking line search reached')
        else:
            eta *= betaparam
            iter += 1
    return eta



#now we can do the gradient descent algorithm

def graddescent(beta_init, lambduh, eta_init, maxiter, x=x_train, y=y_train):
    beta = beta_init
    grad_beta = computegrad(beta, lambduh, x=x, y=y)
    beta_vals = beta
    iter = 0
    while iter < maxiter:
        eta = bt_line_search(beta, lambduh, eta=eta_init, x=x, y=y)
        beta = beta - eta*grad_beta
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta))
        grad_beta = computegrad(beta, lambduh, x=x, y=y)
        iter += 1
        if iter % 100 == 0:
            print('Gradient descent iteration', iter)
    return beta_vals



#lets initialize our lambda, beta and theta,eta and set max iteratoins to 100
lambduh = 0.1
beta_init = np.zeros(d)
theta_init = np.zeros(d)
eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+lambduh)
maxiter = 100

#lets see our gradient calculation. We only need to iterate to 100 to optimize the objective function

betas_grad = graddescent(beta_init, lambduh, eta_init, maxiter)


# We can just use the 100th iteration since we know out gradient will be the local minimm
print(objective(betas_grad[-1, :], lambduh))

#Lets measure the accuracy of or Logistic Regression model by calculating the misscassification error
def compute_misclassification_error(beta, x, y):
    y_pred = 1/(1+np.exp(-x.dot(beta))) > 0.5
    y_pred = y_pred*2 - 1
    return np.mean(y_pred != y)


#lets see how the misclassification error does with our training set. It should be
#small
print(compute_misclassification_error(betas_grad[-1, :], x_train, y_train))


#lets now do it with the test set.
misclasserror = compute_misclassification_error(betas_grad[-1, :], x_test, y_test)
print(misclasserror)

#lets compare this with sklearn's logistic regression
logreg = sklearn.linear_model.LogisticRegression(penalty='l2')

#train the model
trained = logreg.fit(x_train,y_train)



print(classification_report(trained.predict(x_test),y_test))

#lets put them on the same format to measure their performance
#lets comapre the misclassification of the sklearn model with out own
print("Our accuracy is :",1-misclasserror)

