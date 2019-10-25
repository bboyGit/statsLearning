import pandas as pd
import numpy as np

def perceptron(x, y, init, alpha, max_iter, method='original'):
    """
    Desc: Execute perceptron algorithm to do binary classification
    Parameters:
      x: An ndarray. Each row is an observation
      y: An column vector with only 2 categories (-1, +1). Each row represents an response
      init: An column vector contain the initial guess of w and b.
      alpha: An float representing learning rate used in stochastic gradient descent
      max_iter: An int indicating the maximum iteration times
      method: An str either 'dual' or 'original'
    Return: A tuple containing the coefficients of perception function and iterate times.
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

    if method == 'original':
        wb = init.copy()
        one = np.ones([N, 1])
        x_ = np.concatenate([x, one], axis=1)
        f_loss = lambda y_i, x_i, w: y_i * (x_i @ w)[0, 0]
        while count < max_iter:
            count += 1
            Loss = pd.Series([f_loss(y[i, 0], x_[[i], :], wb) for i in range(N)])
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
    elif method == 'dual':
        n = np.zeros(N)
        b = 0
        Gram = x @ x.T
        f_loss = lambda i, y_i, Gram, n, b: y_i * (b + alpha * sum([n[j] * y[j, 0] * Gram[i, j] for j in range(N)]))
        while count < max_iter:
            count += 1
            Loss = pd.Series([f_loss(i, y[i, 0], Gram, n, b) for i in range(N)])
            mistake_idx = Loss[Loss <= 10**(-10)].index
            if len(mistake_idx) == 0:
                break
            else:
                idx = mistake_idx[0]
                the_y = y[idx, 0]
                loss = the_y * (b + alpha * sum([n[j] * y[j, 0] * Gram[idx, j] for j in range(N)]))
                while loss <= 10**(-10):
                    b = b + alpha * the_y
                    n[idx] += 1
                    loss = the_y * (b + alpha * sum([n[j] * y[j, 0] * Gram[idx, j] for j in range(N)]))
        ny = n.reshape(N, 1) * y
        w = alpha * x.T @ ny
        wb = np.concatenate([w, np.array([[b]])])

    return wb, count

if __name__ == "__main__":
    x = np.array([[1, 2], [1.1, 2.1], [0.8, 1.5]])
    y = np.array([[1, 1, -1]]).T
    guess = np.zeros([1, 3]).T
    percep = perceptron(x, y, init=guess, alpha=0.02, max_iter=1000, method='dual')
    wb = percep[0].ravel()
    percep1 = perceptron(x, y, init=guess, alpha=0.02, max_iter=1000, method='original')
    wb1 = percep1[0].ravel()             # wb1与wb相等，可看出原始型感知机与其对偶型算法结果相同
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot(np.linspace(0.7, 1.2, 100), -wb[2]/wb[1] - wb[0]/wb[1] * np.linspace(0.7, 1.2, 100))
