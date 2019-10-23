import pandas as pd
import numpy as np

def perceptron(x, y, init, alpha, max_iter, method='dual'):
    """
    Desc: Execute perceptron algorithm to do binary classification
    Parameters:
      x: An ndarray. Each row is an observation
      y: An column vector with only 2 categories (-1, +1). Each row represents an response
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
    if init.ndim != 2 and init.shape[1] != 1:
        raise Exception('init must be a column vector')
    count = 0
    N = x.shape[0]
    one = np.ones([N, 1])
    x_ = np.concatenate([x, one], axis=1)
    wb = init.copy()

    while count < max_iter:
        count += 1
        Loss = pd.Series([y[i, 0] * (x_[[i], :] @ wb)[0, 0] for i in range(N)])
        mistake_idx = Loss[Loss <= 10**(-10)].index
        if len(mistake_idx) == 0:
            break
        else:
            idx = mistake_idx[0]
            the_x = x_[[idx], :].copy()                        # 选取要进行梯度下降的误分类样本点
            the_y = y[idx, 0]
            loss = the_y * (the_x @ wb)
            while loss <= 10**(-10):
                wb = wb + alpha * the_y * the_x.T
                loss = the_y * (the_x @ wb)
    print(count, Loss)
    converge = False if count >= max_iter else True
    return wb, converge

if __name__ == "__main__":
    x = np.array([[1, 2], [1.1, 2.1], [0.8, 1.5]])
    y = np.array([[1, 1, -1]]).T
    guess = np.zeros([1, 3]).T
    perceptron(x, y, init=guess, alpha=0.02, max_iter=1000)
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot(np.linspace(0.7, 1.2, 100), 1.6 - 0.08 * np.linspace(0.7, 1.2, 100))
