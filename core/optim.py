
import pandas as pd
import numpy as np

class optimize:

    def grad(self, f, x, delta):
        """
        Desc: Get gradient vector of function f in x0
        Parameters:
          f: A function
          x: A columns vector representing vector of x
          delta: A float indicating delta of x
        Return: A vector containing partial derivatives of each elements
        formula: df(x)/dx = (f(x+h) - f(x))/h 当h很小时，约等于导数
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

    def hess(self, f, x, delta):
        """
        Desc: compute Hessian matrix
        formula: df(x, y)/dxdy = (f(x+h, y+h) - f(x+h, y) - f(x, y+h) + f(x, y))/h^2  当h很小时，约等于二阶导数
        """
        fun = f(x)
        nrow = x.shape[0]
        hess_ = np.ones([nrow, nrow])
        for i in range(nrow):
            for j in range(nrow):
                x1 = x.copy()
                x2 = x.copy()
                x3 = x.copy()
                x1[i, 0] = x1[i, 0] + delta
                x2[j, 0] = x2[j, 0] + delta
                if i == j:
                    x3[i, 0] = x3[i, 0] + 2*delta
                else:
                    x3[[i, j], 0] = x3[[i, j], 0] + delta
                f1 = f(x1)
                f2 = f(x2)
                f3 = f(x3)
                hess_[i, j] = (f3 - f2 - f1 + fun)/delta**2
        return hess_

    def grad_desc(self, func, x0, tol, max_iter, alpha, delta):
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
            gradient = self.grad(f=func, x=x_now, delta=delta)          # Compute gradient of given function in point x_now
            diff = np.sqrt(np.sum(gradient**2))
            x_now = x_now - alpha * gradient                          # update point in minus gradient direction
            fun.append(func(x_now))
            count += 1
        success = True if count < max_iter else False
        result = {'x': x_now, 'iter_num': count, 'diff': diff, 'success': success}
        return result

    def newton_iter(self, func, x0, tol, max_iter, delta):
        """
        Desc: get optimize solution by newton iterative method
        """
        count = 0
        x_now = x0.copy()
        while count <= max_iter:
            gradient = self.grad(f=func, x=x_now, delta=delta)
            diff = np.sqrt(np.sum(gradient**2))
            if diff <= tol:
                break
            else:
                count += 1
                hessian_matrix = self.hess(f=func, x=x_now, delta=delta)
                x_now = x_now - np.linalg.inv(hessian_matrix) @ gradient
        success = True if count < max_iter else False
        result = {"x": x_now, 'iter_num': count, 'diff': diff, 'success': success}
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
    optim = optimize()
    Gradient_solution = optim.grad_desc(f, np.zeros([n, 1]), tol=10**(-6), max_iter=10**4, alpha=0.01, delta=10**(-5))
    Newton_solution = optim.newton_iter(f, np.zeros([n, 1]), tol=10**(-6), max_iter=10**4, delta=10**(-5))
    import statsmodels.api as sm
    sm.OLS(y, X).fit().params
