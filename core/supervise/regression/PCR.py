from core.unsupervise.dimReduction.PCA import pca
from core.supervise.regression.ols import ols
import numpy as np

def pcr(x, y, k, const=True):
    """
    Desc: Principal component regression
    Parameters:
      x:  A matrix contain predictors.
      y: A columns vector contain dependent variable
      k: An int indicating the number of component
      const: A bool indicating add const or not
    Return: A dict contain coefficients, fitted values and residual
    Note:
         Although the principal component analysis can perform dimension reduction well. It is still a unsupervised way.
         Cause it doesn't think about the correlation between X and Y, we can not guarantee anything of the ability to
         predict Y by those principal component of X.
    """
    pc = pca(x, k)
    comp = pc[0]
    result = ols(comp, y, const)

    return result
