import numpy as np

def ols(x, y, const=True):
    """
    Desc: Do linear regression by OLS method.
    Parameters:
      x: A matrix contain explanatory variables
      y: A column vector contain dependent variable
      const: A bool, indicating whether we add constant or not. Default True, which means add constant.
    Return: A dict contain fitted values, coefficients and errors.
    """

    # (1) Handle Exceptions
    type_array = type(np.array([[0]]))
    if not isinstance(x, type_array) or not isinstance(y, type_array):
        raise TypeError("Both x and y must be array.")
    if x.ndim != 2 or y.ndim != 2:
        raise Exception("Dimension of x and y must be 2.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The number of observations of x and y must be the same.")

    # (2) Calculate coefficients, errors and fitted values.
    if const:
        c = np.array([[1] * x.shape[0]]).T
        x = np.concatenate([c, x], axis=1)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise Exception("Columns of x are linearly dependent.")

    mat1 = np.linalg.inv(x.T @ x) @ x.T
    beta = mat1 @ y
    y_hat = x @ beta
    error = y - y_hat

    # (3) Tidy results
    result = {'coef': beta, 'fit_value': y_hat, 'resid': error}

    return result


if __name__ == '__main__':
    y = np.array([[2, 1, 4, 3, 5]]).T
    x = np.array([[1, 2.5, 3, 5, 4]]).T
    fit = ols(x, y, const=True)
