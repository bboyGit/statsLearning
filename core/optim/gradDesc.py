import pandas as pd
import numpy as np

def grad(f, x, delta):
    """
    Desc: Get gradient vector of function f in x0
    Parameters:
      f: A function
      x: A columns vector representing vector of x
      delta: A float indicating delta of x
    Return: A column vector containing partial derivatives of each elements
    """
    fun = f(x)
    nrow, ncol = x.shape
    grad_ = np.ones([nrow, ncol])
    for i in range(ncol):
        for j in range(nrow):
            x1 = x.copy()
            x1[j, i] = x1[j, i] + delta
            grad_[j, i] = f(x1)
    grad_ = (grad_ - fun)/delta
    return grad_

def grad_desc(func, x0, tol, max_iter, alpha):
    """
    Desc: Execute Gradient descent algorithm to find local optimal solution of a given function
    Parameters:
      func: A function whose return value is a scalar
      x0: initial guess(columns vector)
      tol: tolerance
      max_iter: max iteration times
      alpha: A positive number (learning rate)
    Return: optimal solution
    """
    count = 0
    diff = 10
    x_now = x0.copy()
    fun = []
    while count < max_iter and diff > tol:
        gradient = grad(f=func, x=x_now, delta=10**(-4))          # Compute gradient of given function in point x_now
        diff = np.sqrt(np.sum(gradient**2))
        x_now = x_now - alpha * gradient                          # update point in minus gradient direction
        fun.append(func(x_now))
        count += 1
    result = {'x': x_now, 'iter_num': count, 'diff': diff, 'fun': fun}
    return result


if __name__ == '__main__':
    # Solve linear regression by gradient descent
    from sklearn.datasets import load_boston
    from core.normalize import normalize
    boston = load_boston()
    y = boston['target'][:100].reshape(100, 1)
    X = boston['data'][:100, :3]
    X = normalize(X).values
    n = X.shape[1]
    f = lambda x: np.dot((y - X @ x).T, y - X @ x)[0, 0]
    solution = grad_desc(f, np.zeros([n, 1]), tol=10**(-6), max_iter=10**4, alpha=0.001)
    solution['x'].ravel()
    solution['iter_num']
    import statsmodels.api as sm
    sm.OLS(y, X).fit().params
