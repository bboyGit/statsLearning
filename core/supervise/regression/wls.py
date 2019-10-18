import numpy as np
from core.supervise.regression.ols import ols

def wls(x, y, const=True, weight=None):
    """
    Desc: Do linear regression by Weighted least square
    Parameters:
      x: A matrix contain predictors
      y: A column vector contain dependent variable
      const: A bool, indicating whether we add constant or not. Default True, which means add constant.
      weight: An 1 dimensional array. Default None.
    Return: A dict contain fitted values, coefficients, errors, weight.
    Note:
        Cost function of WLS: sum(w_i * (y_i - x_i @ beta)^2) <==> sum((sqrt(w_i) * y_i - sqrt(w_i) * x_i @beta)^2)
          <==> (sqrt(W) @ Y - sqrt(W) @ X @ beta).T @ (sqrt(W) @ Y - sqrt(W) @ X @ beta)
          Where W is a diagonal matrix with W[i, i] = w_i
        So, we can see that WLS of Y ~ X with weight W is identical to OLS of sqrt(W) @ Y ~ sqrt(W) @ X.
    """
    # (1) Handle Exceptions
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Both x and y must be array.")
    if x.ndim != 2 or y.ndim != 2:
        raise Exception("Dimension of x and y must be 2.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The number of observations of x and y must be the same.")

    # (2) Deal weight
    if weight == None:
        # 用户不填权重时我们默认用残差绝对值的倒数作为权重
        ols_fit = ols(x, y, const)
        error = np.abs(ols_fit['resid'][:, 0])
        weight = np.diag(error)
        weight_sqrt = np.sqrt(weight)
    elif isinstance(weight, np.ndarray):
        if (weight < 0).any():
            raise Exception("Negative weight is forbidden")
        elif weight.ndim != 1:
            raise Exception("The dimension of weight must be 1")
        else:
            weight = np.diag(weight)
            weight_sqrt = np.sqrt(weight)
    else:
        raise TypeError("Type of weight must be array")

    # (3) Calculate coefficient, fitted value and error
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
    error = y - y_hat

    # (3) Tidy results
    result = {'coef': beta, 'fit_value': y_hat, 'resid': error, 'weight': weight}

    return result

if __name__ == "__main__":
    y = np.array([[2, 1, 4, 3, 5, 3.27, 4.52]]).T
    x = np.array([[1, 2.5, 3, 5, 4, 3.6, 5.3]]).T
    fit1 = wls(x, y, const=True)