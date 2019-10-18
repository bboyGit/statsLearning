
import numpy as np

def local_ls(x, y, k, center=None, const=True):
    """
    Desc: Execute the Local weighted least square linear regression
    Parameters:
      x: A matrix contain explanatory variables. m row and n col means m observation and n features.
      y: A column vector contain dependent variable
      k: A float indicating the tuning parameter of weight. The smaller the k, the more flexible the model.
      center: A row vector indicating a point. Default None.
      const: A bool, indicating whether we add constant or not. Default True, which means add constant.
    Return: A dict contain fitted value
    Note:
        Local weighted least square is a non-parametric learning algorithm. So, each time you wanna predict you have to
        fit the model again, which mean you have to save train data after fit the model.
    """
    # (1) Deal exception
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Both x and y must be array.")
    if x.ndim != 2 or y.ndim != 2:
        raise Exception("Dimension of x and y must be 2.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The number of observations of x and y must be the same.")

    # (2) Fit model
    nobs = x.shape[0]
    f = lambda xi, x0, k: np.exp(- ((xi - x0) @ (xi - x0).T)[0, 0] / (2 * k**2))
    if not isinstance(center, np.ndarray):
        # which means we should get the fit value of the train set
        y_hat = []
        for i in range(nobs):
            x0 = x[[i], :]
            weight = np.array([f(x[[j], :], x0, k) for j in range(nobs)])
            weight = np.diag(weight)
            weight_sqrt = np.sqrt(weight)
            if np.linalg.matrix_rank(x) < x.shape[1]:
                raise Exception("Columns of x are linearly dependent.")
            if const:
                c = np.array([[1] * x.shape[0]]).T
                x1 = np.concatenate([c, x], axis=1)
                if np.linalg.matrix_rank(x1) < x1.shape[1]:
                    raise Exception("Columns of x are linearly dependent.")
            X = weight_sqrt @ x1
            Y = weight_sqrt @ y
            mat1 = np.linalg.inv(X.T @ X) @ X.T
            beta = mat1 @ Y
            the_y_hat = x1 @ beta
            y_hat.append(the_y_hat[i])
        y_hat = np.array(y_hat)

        return y_hat
    else:
        # which means we'll predict textY
        weight = np.array([f(x[[j], :], center, k) for j in range(nobs)])
        weight = np.diag(weight)
        weight_sqrt = np.sqrt(weight)
        if const:
            c = np.array([[1] * x.shape[0]]).T
            x = np.concatenate([c, x], axis=1)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise Exception("Columns of x are linearly dependent.")
        X = weight_sqrt @ x
        Y = weight_sqrt @ y
        mat1 = np.linalg.inv(X.T @ X) @ X.T
        beta = mat1 @ Y
        y_hat = x @ beta
        center_y = np.concatenate([np.array([[1]]), center], axis=1) @ beta

        return center_y, center, beta, y_hat


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import matplotlib.pyplot as plt
    boston = load_boston()
    y = boston['target'][:500].reshape(500, 1)
    x = boston['data'][:500, 7].reshape(500, 1)
    fit_local = local_ls(x, y, k=0.1, const=True)
    fit_ = local_ls(x, y, k=0.1, center=np.array([[5]]), const=True)
    
    # Compare OLS and local regression by figure
    from core.supervise.regression.OLS import ols
    fit = ols(x, y)
    x_ = np.linspace(4.8, 5.2, 100)
    plt.plot(x_, fit_[2][0, 0] + fit_[2][1, 0] * x_)
    plt.plot(x, fit['fit_value'])
    plt.scatter(x, y, s=5)
    x_fit = np.concatenate([x, fit_local], axis=1)
    x_fit = x_fit[x_fit[:, 0].argsort(), :]
    plt.plot(x_fit[:, 0], x_fit[:, 1])
    plt.legend(['local_regression 5', 'OLS', 'x-y', 'local_regression'])
