
import numpy as np

def r_square(y, y_hat, p, adjust = True):
    """
    Desc: Compute R square.
    Parameter:
      y: The true value of y
      y_hat: The fitted value of y
      p: The number of features of the learning model
      adjust: A bool. If it's True, we calculate adjusted R square.
    Return: A float
    """
    tss = np.sum((y - y.mean())**2)
    rss = np.sum((y - y_hat)**2)
    if adjust:
        n = len(y)
        adj = (n - 1)/(n - p - 1)
        r2 = 1 - adj * rss/tss
    else:
        r2 = 1 - rss/tss

    return r2

def aic(y, y_hat, p):
    """
    Desc: Compute AIC information criterion
    """
    rss = np.sum((y - y_hat)**2)
    var = (y - y_hat).var()
    n = len(y)
    AIC = (rss + 2 * p * var)/(n * var)
    return AIC

def bic(y, y_hat, p):
    """
    Desc: Compute BIC information criterion
    """
    rss = np.sum((y - y_hat) ** 2)
    var = (y - y_hat).var()
    n = len(y)
    BIC = (rss + np.log(n) * p * var)/(n * var)

    return BIC


if __name__ == "__main__":
    from core.supervise.regression.ols import ols
    y = np.array([[2, 1, 4, 3, 5]]).T
    x = np.array([[1, 2.5, 3, 5, 4]]).T
    fit = ols(x, y, const=True)
    aic(y, fit['fit_value'], 1)
