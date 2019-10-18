
import pandas as pd
import numpy as np

def spline(x, y, power, knot, const=True):
    """
    Desc: Execute regression spline algorithm
    Parameters:
      x: A 1-D vector
      y: A columns vector
      power: An int representing the power of spline
      knot: Optional. If it's an int, it represents the number of knots. If it's an 1-D array, then it's cutting points.
    Return: fitted values.
    Note:
        function form:
            f(x) = a + a1*x + a2*x^2 + a3*x^3 + sum of g(x, K_i)
            where g(x, K_i) = (x - K_i)^knot if x > K_i else 0
        properties: Function is continuous in each cutting point in its 0, 1, 2...... knot - 1 derivatives.
    """
    # (1) Deal exception
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Both x and y must be array.")
    if x.ndim != 1:
        x = x.ravel()
    if x.shape[0] != y.shape[0]:
        raise Exception("The number of observations of x and y must be the same.")

    # (2) Construct X
    X = {}
    for i in range(power):
        X[i] = x ** (i + 1)
    if isinstance(knot, np.ndarray):
        # knot represents cutting points for spline
        for idx, value in enumerate(knot):
            X[idx + 3] = [(i - value)**power if i > value else 0 for i in x]
    elif isinstance(knot, int):
        # knot represents the number of cutting points
        if knot > len(x):
            raise Exception("The number of knot can not exceed the number of observations")
        x_max, x_min = x.max(), x.min()
        knot1 = x_min + np.linspace(0, x_max - x_min, knot + 2)[1: -1]                # 等距离设置cutting points
        for idx, value in enumerate(knot1):
            X[idx + power] = [(i - value)**power if i > value else 0 for i in x]

    X = pd.DataFrame(X).values

    # (3) Fit model
    if const:
        c = np.array([[1] * x.shape[0]]).T
        X = np.concatenate([c, X], axis=1)
    mat1 = np.linalg.inv(X.T @ X) @ X.T
    beta = mat1 @ y
    y_hat = X @ beta

    return y_hat

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import matplotlib.pyplot as plt
    boston = load_boston()
    xy = np.concatenate([boston['data'][:500, 5].reshape(500, 1), boston['target'][:500].reshape(500, 1)], axis=1)
    xy = xy[xy[:, 0].argsort(), :]
    x = xy[:, 0].reshape(500, 1)
    y = xy[:, 1].reshape(500, 1)
    spline1 = spline(x, y, power=1, knot=5, const=True)
    spline3 = spline(x, y, power=3, knot=5, const=True)

    from core.supervise.regression.OLS import ols
    fit = ols(x, y)
    plt.plot(x, fit['fit_value'])
    plt.plot(x, spline1)
    plt.plot(x, spline3)
    plt.scatter(x, y, s=5)
    plt.legend(['OLS', 'spline1', 'spline3', 'x-y'])
    plt.xlabel('average number of rooms per dwelling')
    plt.ylabel('house price')






