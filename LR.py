# implementation of logistic regression 
# using sparse matrix to reduce computation time and memory usage
import numpy as np
from scipy.linalg import blas
from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


class LogisticRegression_():
    def __init__(self, alpha=0.1, max_iter =150):
        self.w = None
        self.history = None
        self.max_iter = max_iter
        self.alpha = alpha

    def sigmoid(self, x):
        # activation function used for mapping
        y = expit(x)
        return y

    def weighted_input(self, x, beta):
        # calculate the weighted inputs

        result = x.dot(beta)
        return result

    def calculate_p(self, x, beta):
        # calculate the probability using the sigmoid
        
        y = self.weighted_input(x, beta)
        result = self.sigmoid(y)
        return result

    def cost_func(self, x, y, beta):
        m = x.shape[0] # number of rows
        n = x.shape[1] # number of columns
        
        # calculate the cost function of training samples

        p = self.calculate_p(x, beta)

        err = (y * np.log(p)) + ((1 - y) * np.log(1-p))
        
        sum_err = np.sum(err)
        total_cost = -(1/m) * sum_err
        return total_cost

    def get_gradient(self, x, y, beta):
        # calculate the gradient
        m = x.shape[0] # number of rows
        n = x.shape[1] # number of columns
        
        x_t = x.transpose()
        p = self.calculate_p(x, beta)
        y_ = np.subtract(p, y)
        dot_ = x_t.dot(y_)
        result = (1 / m) * dot_

        return result
    
    def normalization(self, x):
        scaler = StandardScaler(with_mean=False)
        scaler.fit(x)
        mean = scaler.mean_
        std = scaler.scale_
        x_norm = (x-mean)/std
        return x_norm
        
    def gradient_descent(self, x, y, beta, alpha, max_iter):
        """
        using gradient descent to find the optimal beta which minimize the cost function

        :param x: array-like (samples)
        :param y: array-like (targeted values or labels)
        :param beta: array-like (weights)
        :param alpha: learning rate
        :param max_iter: the maximum iteration for the algorithm
        :return: the optimal beta and the history
        """
        
        m = x.shape[0] # number of rows
        n = x.shape[1] # number of columns
        cost_history = []
        for i in range(max_iter):
            cost = self.cost_func(x, y.reshape(m, 1), beta)
            grad = self.get_gradient(x, y.reshape(m, 1), beta)

            beta = beta - (alpha * grad)

            cost_history.append(cost)

        return beta, cost_history
    
    
    def fit(self, x, y):
        # function that fit the model
        m = x.shape[0] # number of rows
        n = x.shape[1] # number of columns
        y = y.reshape((m,1))
        x = sparse.csr_matrix(x)

        """
        if possible, do feature normalization before fitting the model
        """

        if x.mean() > 1: # feature normalization if not normalized
            x = self.normalization(x)
        inital_beta = np.zeros((n + 1, 1)) # initializing beta
        temp = sparse.csr_matrix(np.ones(m).reshape(m,1))
        x = sparse.hstack((temp, x)) # adding the intercept
        opt_beta = self.gradient_descent(x, y, inital_beta, self.alpha, self.max_iter) # using the gradient descent to find the optimal beta
        self.w = opt_beta[0]
        self.history = opt_beta[1]
        return self
    
    def calculate_z(self, x):
        # function used to calculate the z value which is used to calculate the probabilities of 2 labels
        beta = self.w
        x = self.normalization(x)
        m = x.shape[0] # number of rows
        n = x.shape[1] # number of columns
        temp = sparse.csr_matrix(np.ones(m).reshape(m,1))
        x = sparse.hstack((temp, x))
        result = x.dot(beta)
        return result
    
    def predict_prob(self, x):
        # function that calculated the probabilities for two classes
        z = self.calculate_z(x)
        p_ = np.exp(z)
        p_1 = p_ / (1+p_)
        p_0 = 1 - p_1
        p = np.column_stack((p_0, p_1))
        return p

    def predict(self, x):
        # function that predict the class
        prob = self.predict_prob(x)
        pred_class = []
        for pred_p in prob:
            pred_class.append(np.argmax(pred_p))
        return pred_class
