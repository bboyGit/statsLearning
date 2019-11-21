
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
        self.x = normalize(x, method='max_min')
        self.x.dropna(axis=1, inplace=True)
        self.x = self.x.values
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

    def __binary(self, x, y):
        """
        Desc: Fit binary logistic regression
        :return: coefficients of binary logistic regression
        """
        # (1) create loss function
        N, n = x.shape
        sigmoid = lambda z: 1/(1 + np.exp(-z))
        loss1 = lambda w: (y.T @ np.log(sigmoid(x @ w)))[0, 0]
        loss2 = lambda w: ((1 - y.T) @ np.log(1 - sigmoid(x @ w)))[0, 0]
        loss = lambda w: -(loss1(w) + loss2(w))/N
        # (2) use gradient descent to find optimal coefficient
        opt = grad_desc(loss, self.x0, tol=self.tol, max_iter=self.max_iter, alpha=self.alpha)
        coef = opt['x']
        iter_num = opt['iter_num']
        loss_val = opt['fun'][-1]
        fit_val = sigmoid(x @ coef)
        return coef, iter_num, loss_val, fit_val

    def __IIA(self, y_uniq):
        """
        Desc: Fit multi-class logistic regression in the assumption of IIA(independence of irrelevant alternatives)
        """
        main_class = y_uniq[0]
        remain_class = y_uniq[1:]
        xy = np.concatenate([self.y, self.x], axis=1)
        for i in remain_class:
            sub_xy = pd.DataFrame(xy)[pd.DataFrame(xy).isin([i, main_class]).iloc[:, 0]]
            sub_xy = sub_xy.values.copy()
            the_y = [0 if j == main_class else 1 for j in sub_xy[:, 0]]
            the_y = np.array([the_y]).T
            the_x = sub_xy[:, 1:]
            coef_i, iter_num_i, loss_val_i, fit_val_i = self.__binary(the_x, the_y)
            if 'coef' not in locals():
                coef = coef_i
                iter_num = [iter_num_i]
                loss_val = [loss_val_i]
            else:
                coef = np.concatenate([coef, coef_i], axis=1)
                iter_num.append(iter_num_i)
                loss_val.append(loss_val_i)
        e_wx = np.exp(self.x @ coef)              # e_wx 的第i列是 exp(x @ coef_i)
        div = np.array([1 + np.sum(e_wx, axis=1)]).T
        fit_val = [1/div if i == 0 else e_wx[:, [i-1]]/div for i in range(len(y_uniq))]
        fit_val = np.concatenate(fit_val, axis=1).round(4)
        return coef, iter_num, loss_val, fit_val

    def __ovr(self, y_uniq):
        """
        Desc: Fit multi-class logistic regression by one vs result method
        """
        for response in y_uniq:
            the_y = self.y.copy()
            the_y = [1 if i == response else 0 for i in the_y.ravel()]
            the_y = np.array([the_y]).T
            coef_i, iter_num_i, loss_val_i, fit_val_i = self.__binary(self.x, the_y)   # call binary logistic regression
            if 'coef' not in locals():
                coef = coef_i
                iter_num = [iter_num_i]
                loss_val = [loss_val_i]
                fit_val = fit_val_i
            else:
                coef = np.concatenate([coef, coef_i], axis=1)
                iter_num.append(iter_num_i)
                loss_val.append(loss_val_i)
                fit_val = np.concatenate([fit_val, fit_val_i], axis=1)
        fit_val = fit_val.round(4)
        return coef, iter_num, loss_val, fit_val

    def __softmax(self, x, y, y_uniq):
        """
        Desc: Fit multi-class logistic regression by softmax function
        """
        # (1) create loss function
        N, n = x.shape
        K = len(y_uniq)
        x0 = np.zeros([n, K])                                                                # initial guess
        softmax = lambda x_i, _w, y_i: np.exp(x_i @ _w[:, [y_i]])/np.exp(x_i @ _w).sum()     # z is x@w(a N by K matrix)
        loss = lambda w: -np.sum([np.log(softmax(x[[i], :], w, y[i, 0])) for i in range(N)])/N
        # (2) find optimal coefficient by gradient descent
        opt = grad_desc(loss, x0, self.tol, self.max_iter, self.alpha)
        coef = opt['x']
        iter_num = opt['iter_num']
        loss_val = opt['fun'][-1]
        fit_val = np.exp(x @ coef)/np.sum(np.exp(x @ coef), axis=1)
        return coef, iter_num, loss_val, fit_val

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
            result = self.__binary(self.x, self.y)
        else:
            if method == 'IIA':
                result = self.__IIA(y_uniq)
            elif method == 'ovr':
                result = self.__ovr(y_uniq)
            elif method == 'softmax':
                result = self.__softmax(self.x, self.y, y_uniq)
        return result

if __name__=='__main__':
    import sklearn.datasets as dataset
    from sklearn import linear_model
    from datetime import datetime
    # multi-class
    digit = dataset.load_digits()
    x = digit['data']
    y = digit['target']
    y = y.reshape(len(y), 1)
    start = datetime.now()
    logit1 = logistic(x=x, y=y, tol=10**(-5), max_iter=20, alpha=10**(-1), x0=None)
    result1 = logit1.fit('softmax')
    end = datetime.now()
    fit = linear_model.LogisticRegression(fit_intercept=True,
                                          solver='newton-cg', penalty='none').fit(x, y.ravel())
    np.concatenate([fit.intercept_, fit.coef_.ravel()])
