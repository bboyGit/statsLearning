
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class svm:
    """
    Desc: It's a class who can achieve linearly dividable support vector machine, linear support vector machine
        and non-linear support vector machine
    """
    def __init__(self, feature, response, dual, kenerl, max_iter=1000, method='linear', C=1, dividable=False):
        """
        :param feature: Array or DataFrame.
        :param response: Array or DataFrame(only 1 column)
        :param dual: Bool. if it's True, we use dual algorithm. Otherwise, we use original algorithm
        :param method: Str, linear_dividable, linear or non_linear.
        :param kenerl: Str.
        :param max_iter: Int.
        :param C: float. Regularize coefficient
        :param dividable: bool, 给出一个数据是否线性可分的预判断。若为True，则按线性可分的方法求解，否则按线性不可分的放式求解。
        """
        self.feature = self.check_type(feature)
        self.response = self.check_type(response)
        self.dual = dual
        self.method = method
        self.kenerl = kenerl
        self.max_iter = max_iter
        self.dividable = dividable
        if C < 0:
            raise Exception('C must be positive')
        else:
            self.C = C

    def check_type(self, data):
        if isinstance(data, np.ndarray):
            result = data
        elif isinstance(data, pd.DataFrame):
            result = data.values
        elif isinstance(data, pd.Series):
            result = data.values
        else:
            raise TypeError('feature must be np.ndarray or pd.DataFrame')
        return result

    def check_adj_res(self):
        y_uniq = np.unique(self.response)
        if len(y_uniq) > 2:
            raise Exception('The number of class larger than 2')
        elif len(y_uniq) in [0, 1]:
            raise Exception('The number of class is {}'.format(len(y_uniq)))
        else:
            idx1 = np.where(self.response == y_uniq[0])[0]
            idx2 = np.where(self.response == y_uniq[1])[0]
            self.response[idx1] = 1
            self.response[idx2] = -1

    def __linear(self):
        """
        Desc: 拟合线性支持向量机，得到分离超平面 w.T @ x + b = 0.
        optimize problem:
          线性可分时：
              original:
                min: (w.T @ w)/2
                st: y_i * (w.T @ x_i + b) - 1 >= 0 for i = 1,...,N
              dual:
                min:  1/2 * sum(a.T @ ((y * x) @ (y.T * x.T)) @ a) - sum(a)
                st: a_i >= 0 for i = 1,...,N  & sum(a * y) = 0
                其中，a是拉格朗日函数中 1 - y_i * (w.T @ x_i + b) 的系数
          线性不可分时：
              original:
                min: (w.T @ w)/2 + C * sum(slack_variable)
                st: y_i * (w.T @ x_i + b) + slack_variable_i - 1 >= 0 for i = 1,...,N
              dual:
                min:  1/2 * sum(a.T @ ((y * x) @ (y.T * x.T)) @ a) - sum(a)
                st: 0 <= a_i <= C for i = 1,...,N  & sum(a * y) = 0
        """
        N, n = self.feature.shape
        x = self.feature.copy()
        y = self.response.copy()
        y = y.reshape(len(y), 1)
        # 构造原始线性支持向量机的优化模型
        p = (y * x) @ (y * x).T
        def loss(a):
            a = a.reshape(N, 1)                # np.array([[2, 1]]).T * np.array([[1,2], [3,4]]) 是2*[1,2] 以及 1*[3, 4]
            loss_val = (1/2 * a.T @ p @ a - np.sum(a))[0, 0]
            print(loss_val)
            return loss_val
        cons = [{'type': 'eq', 'fun': lambda a: np.sum(a * y)}]
        if self.dividable:
            bound = [(0, np.inf) for i in range(N)]
        else:
            bound = [(0, self.C) for i in range(N)]
        res = []
        loss_val = []
        # 做100次优化，取其中优化结果最好的结果。
        for i in range(100):
            num = np.random.uniform(-10, 10, n)
            low_num = num.min()
            high_num = num.max()
            a = np.random.uniform(low_num, high_num, len(y))
            res1 = minimize(fun=loss, x0=a, method='SLSQP', constraints=cons, bounds=bound)     # 从目前来看，minimize下的SLSQP方法无法找到svm的最小值
            res.append(res1)
            loss_val.append(res1['fun'])
        best_fit_idx = np.array(loss_val).argmin()
        best_fit = res[best_fit_idx]
        alpha = best_fit['x'].reshape(N, 1)
        w = (alpha * y * x).sum(axis=0)                     # 按列求和
        w = w.reshape(len(w), 1)
        sv_idx = np.where(alpha.ravel() > 0)[0]
        sv = x[sv_idx, :]                                   # 支持向量
        x_j = sv[[0], :]                                    # 选择第一个支持向量计算b
        y_j = y[sv_idx[0], 0]
        b = y_j - x_j @ w
        return alpha, w, b

    def __non_linear(self):
        return

    def fit(self):
        # self.check_adj_res()
        if self.method == "linear":
            result = self.__linear()
        elif self.method == "non_linear":
            result = self.__non_linear()
        else:
            Exception('method must be one of "linear_dividable", "linear" or "non_linear"')
        return result

if __name__=='__main__':
    x = np.array([[1, 2], [2, 2.1], [0.8, 1.5]])
    y = np.array([1, 1, -1])
    _svm = svm(x, y, True, kenerl=None, method='linear')
    result = _svm.fit()
    print(result)
