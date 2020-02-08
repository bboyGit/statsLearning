
import numpy as np
from core.unsupervise.svd import svd

def pca(x, k, scale):
    """
    Desc: Execute principal component analysis to achieve dimensionality reduction
    Parameters:
      x: The input 2D array
      k: An int representing the objective dimension
    Return: The principal component y, explained variance ratio and principal component's contribution ratio to x
    """
    # (1) Make mean of x equal to zero
    if scale:
        x = (x - x.mean(axis=0))/x.std(axis=0)

    # (2) Get eigenvalues and eigen vectors of x.T @ x by svd
    svd_result = svd(x.T @ x, k=None).decompose()
    cov_mat, V = svd_result[1], svd_result[2]

    # (3) Get the explained variance ratio
    var_ratio = cov_mat[:k, :k].diagonal().cumsum()/cov_mat.diagonal().sum()

    # (4) Get the pre kth principal component
    y = x @ V[:, :k]

    # (5) Get contribution ratio of principal component to original variable
    contr_ratio = np.array([np.sum([np.corrcoef(x[:, i], y[:, j])[0, 1]**2 for j in range(k)]) for i in range(x.shape[1])])

    return y, var_ratio, contr_ratio

if __name__ == "__main__":
    mat = np.random.normal(loc=1, scale=2, size=200).reshape([20, 10]).round(2)
    result = pca(mat, 3, scale=True)
    y = result[0]
    print(result)
    print(y.T @ y)
