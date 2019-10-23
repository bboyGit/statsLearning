import pandas as pd
import numpy as np

def perceptron(x, y, init, alpha, max_iter, method='dual'):
    """
    Desc: Execute perceptron algorithm to do binary classification
    Parameters:
      x: An ndarray. Each row is an observation
      y: An column vector with only 2 categories (-1, +1). Each row means an response
      init: An column vector contain the initial guess of w and b.
      alpha: An float representing learning rate used in stochastic gradient descent
      max_iter: An int indicating the maximum iteration times
      method: An str either 'dual' or 'normal'
    Return: The coefficients of perception function
    Note:
        Perceptron is a kind of linear binary classifier which classify samples by a fitted hyperplane(x*w + b = 0).
        And our goal is to find w and b so that all sample points will be correctly classified by x*w + b = 0.
        Only when the input data set are linearly dividable, perceptron can be successfully fitted.
        Model formation:
            y = sign(x*w + b)
        Loss function:
            L(w, b) = - sum([y_i * (x_i * w + b) if y_i * (x_i * w + b) <= 0 else 0 for i in range(N)])
        Optimal algorithm:
            Stochastic gradient descent which means we only consider one sample in one gradient descent process
    """
    count = 0
    N = x.shape[0]
    one = np.ones([N, 1])
    x_ = np.concatenate(x, one)
    wb = init.copy()
    f = lambda x: x @ wb

    while count < max_iter:
        count += 1
        value = pd.Series([y[i, 0] * f(x_[i, :])[0, 0] for i in range(N)])
        mistake_idx = value[value <= 0].index
        if len(mistake_idx) == 0:
            break
        else:
            the_x = x_[mistake_idx[0], :].copy()                # 选取要进行梯度下降的误分类样本点
            the_y = y[mistake_idx, 0]
            loss = the_y * (the_x @ wb)
            while loss <= 0:
                wb[:-1, 0] = wb[:-1, 0] + alpha * the_y * the_x.T
                wb[-1, 0] = wb[-1, 0] + alpha * the_y
                loss = the_y * (the_x @ wb)

    return wb

if __name__ == "__main__":
    pass
