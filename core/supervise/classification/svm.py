
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from warnings import warn

class svm:
    """
    Desc: It's a class who can achieve linearly dividable support vector machine, linear support vector machine
        and non-linear support vector machine
    """
    def __init__(self, feature, response, dual, kenerl, max_iter=1000, method='linear'):
        """
        :param feature: Array or DataFrame.
        :param response: Array or DataFrame(only 1 column)
        :param dual: Bool. if it's True, we use dual algorithm. Otherwise, we use original algorithm
        :param method: Str, linear_dividable, linear or non_linear.
        :param kenerl: Str.
        :param max_iter: Int.
        """
        self.feature = self.check_type(feature)
        self.response = self.check_type(response)
        self.dual = dual
        self.method = method
        self.kenerl = kenerl
        self.max_iter = max_iter

    def check_type(self, data):
        if isinstance(data, np.ndarray):
            result = data
        elif isinstance(data, pd.DataFrame):
            result = data.values
        elif isinstance(data, pd.Series):
            result = data.values.reshape(data.shape[0], 1)
        else:
            raise TypeError('feature must be np.ndarray or pd.DataFrame')
        return result

    def check_adj_res(self):
        y_uniq = np.unique(self.response[:, 0])
        if len(y_uniq) > 2:
            raise Exception('The number of class larger than 2')
        elif len(y_uniq) in [0, 1]:
            raise Exception('The number of class is {}'.format(len(y_uniq)))
        else:
            self.response[self.response == y_uniq[0]] = 1
            self.response[self.response == y_uniq[1]] = -1

    def __linear_divide(self):
        """
        Desc: 拟合线性可分支持向量机，得到分离超平面 w.T @ x + b = 0.
        optimize problem:
          original:
            min: (w.T @ w)/2
            st: y_i * (w.T @ x_i + b) - 1 >= 0 for i = 1,...,N
          dual:
            min:  1/2 * sum(a * y * x @ (a.T * y.T * x.T)) - sum(a)
            st: a_i >= 0 for i = 1,...,N  & sum(a * y) = 0
            其中，a是拉格朗日函数中 1 - y_i * (w.T @ x_i + b) 的系数
        """
        N, n = self.feature.shape
        x = self.feature.copy()
        y = self.response.copy()
        if self.dual:
            # 给出参数的初始值
            a = np.ones([N, 1])
            # 构造原始线性可分支持向量机的优化模型
            def loss(a):
                a = a.reshape(N, 1)
                part1 = np.sum((a * y * x) @ (a.T * y.T * x.T))
                loss_val = 1/2 * part1 - np.sum(a)    # np.array([[2, 1]]).T * np.array([[1,2], [3,4]]) 是2*[1,2] 以及 1*[3, 4]
                return loss_val
            cons = ({'type': 'eq', 'fun': lambda a: (a * y).ravel()})
            bound = [(0, np.inf) for i in range(N)]
            res = minimize(fun=loss, x0=a, method='SLSQP', constraints=cons, bounds=bound)
            if not res['success']:
                warn('optimize failed')
            alpha = res['x'].reshape(N, 1)
            w = (alpha * y * x).sum(axis=0)             # 按列求和
            sv_idx = np.where(alpha.ravel() > 0)[0]

            sv = x[sv_idx, :]                                   # 支持向量
            x_j = sv[[0], :]                                    # 选择第一个支持向量计算b
            y_j = y[sv_idx[0], 0]
            b = y_j - sum((alpha * y * x) @ x_j.T)
        else:
            print('还没想好怎么搞')
        return alpha, w, b

    def __linear(self):
        return

    def __non_linear(self):
        return

    def fit(self):
        self.check_adj_res()
        if self.method == "linear_dividable":
            result = self.__linear_divide()
        elif self.method == "linear":
            result = self.__linear()
        elif self.method == "non_linear":
            result = self.__non_linear()
        else:
            Exception('method must be one of "linear_dividable", "linear" or "non_linear"')
        return result

if __name__=='__main__':
    x = np.array([[1, 2], [1.1, 2.1], [0.8, 1.5]])
    y = np.array([[1, 1, -1]]).T
    _svm = svm(x, y, True, kenerl=None, method='linear_dividable')
    result = _svm.fit()

