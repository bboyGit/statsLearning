
import numpy as np
import pandas as pd
from core.normalize import normalize
from core.optim.gradDesc import grad_desc
from core.add_const import add_const

class logistic:
    """
    Desc: This is a logistic regression object who can fit binary or multi-class logistic regression
    """
    def __init__(self, x, y, tol, max_iter, alpha, x0=None):
        """
        :param x: features in training set
        :param y: response in training set
        :param x0: initial guess
        :param tol: tolerance
        :param max_iter: max iteration times
        :param alpha: A positive number (learning rate)
        """
        self.x = normalize(x, method='max_min').values
        self.x = add_const(self.x)
        self.y = y.copy()
        n = self.x.shape[1]
        if x0 is None:
            self.x0 = np.zeros([n, 1])
        else:
            self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha

    def __binary(self):
        """
        Desc: Fit binary logistic regression
        :return: coefficients of binary logistic regression
        """
        # (1) create loss function
        x = self.x.copy()
        y = self.y.copy()
        N, n = x.shape
        sigmoid = lambda z: 1/(1 + np.exp(-z))
        loss = lambda w: -(y.T @ np.log(sigmoid(x @ w)) + (1 - y.T) @ np.log(1 - sigmoid(x @ w)))[0, 0]/N
        # (2) use gradient descent to find optimal coefficient
        opt = grad_desc(loss, self.x0, tol=self.tol, max_iter=self.max_iter, alpha=self.alpha)
        coef = opt['x'].ravel()
        iter_num = opt['iter_num']
        return coef, iter_num

    def __IIA(self):
        """
        Desc: Fit multi-class logistic regression in the assumption of IIA(independence of irrelevant alternatives)
        """
        return

    def __ovr(self):
        """
        Desc: Fit multi-class logistic regression by one vs result method
        """
        return

    def __ovo(self):
        """
        Desc: Fit multi-class logistic regression by one vs one method
        :return:
        """
        return

    def __softmax(self):
        """
        Desc: Fit multi-class logistic regression by softmax function
        :return:
        """
        return

    def __multiple(self, method):
        """
        Desc: Fit multi-class logistic regression
        :param method: 'ovr', 'ovo', or 'softmax'.
        :return: coefficients of multi-class logistic regression
        """
        if method == 'IIA':
            result = self.__IIA()
        elif method == 'ovr':
            result = self.__ovr()
        elif method == 'ovo':
            result = self.__ovo()
        elif method == 'softmax':
            result = self.__softmax()
        return result

    def fit(self, method=None):
        """
        Desc: Fit binary or multi-class logistic regression
        :param method: Method to fit a multi-class logistic regression.'IIA','ovr', 'ovo', or 'softmax' are candidates.
        :return: coeficients of logistic regression
        """
        y_uniq = np.unique(self.y)
        no_class = len(y_uniq)
        if no_class <= 1:
            raise Exception('Training set has only 1 class')
        elif no_class == 2:
            coef = self.__binary()
        else:
            coef = self.__multiple(method)
        return coef

if __name__=='__main__':
    import sklearn.datasets as dataset
    import matplotlib.pyplot as plt
    breast_cancer = dataset.load_breast_cancer()
    train_x = breast_cancer['data'][:500, :3]
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    logit = logistic(x=train_x, y=train_y, tol=10**(-5), max_iter=10**5, alpha=10**(-3), x0=None)
    coef = logit.fit()
    coef
    from sklearn import linear_model
    fit = linear_model.LogisticRegression(fit_intercept=True,
                                          solver='newton-cg', penalty='none').fit(train_x, train_y.ravel())
    np.concatenate([fit.intercept_, fit.coef_.ravel()])
