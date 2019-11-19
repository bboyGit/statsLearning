
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
    grad_vector = []
    fun = f(x)
    for i, j in enumerate(x):
        x1 = x.copy()
        x1[i, :] = j + delta
        fun1 = f(x1)
        partial_deriv = (fun1 - fun)/delta
        grad_vector.append(partial_deriv)
    grad_vector = np.array([grad_vector]).T
    return grad_vector

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
        diff = np.sum(np.abs(gradient)) * alpha
        x_now = x_now - alpha * gradient                          # update point in minus gradient direction
        fun.append(func(x_now))
        count += 1

    result = {'fun': fun, 'x': x_now, 'iter_num': count}

    return result


if __name__ == '__main__':
    # Solve linear regression by gradient descent
    y = np.array([[2, 1, 5, 9, 7]]).T
    X = np.array([[0.3, 1, 1.3, 0.6],
                  [0.45, 1.1, 1, 0.49],
                  [0.19, 0.89, 0.87, 0.5],
                  [0.8, 1.7, 1, 0.6],
                  [0.56, 0.77, 0.4, 0.77]])
    f = lambda x: np.dot((y - X @ x).T, y - X @ x)[0, 0]
    grad(f, x=np.array([[1, 2.4, 3, 4]], dtype=float).T, delta=0.001)
    solution = grad_desc(f, np.array([[0, 0, 0, 0]], dtype=float).T, tol=10**(-6), max_iter=3 * 10**4, alpha=0.01)
    solution['x'].ravel()
    import statsmodels.api as sm
    sm.OLS(y, X).fit().params
