
import numpy as np

def pca(x, k):
    """
    Desc: Execute principal component analysis to achieve dimensionality reduction
    Parameters:
      x: The input 2D array
      k: An int representing the objective dimension
    Return: The principal component y and explained variance ratio
    """
    # (1) Make mean of x equal to zero
    x = x - x.mean(axis=0)

    # (2) Get eigenvalues and eigen vectors of x.T @ x
    eigval, eigvec = np.linalg.eig(x.T @ x)

    # (3) Reorder eigenvalues and eigen vectors by descending way
    idx = eigval.argsort()
    order_idx = [idx[i] for i in range(len(idx) - 1, -1, -1)]
    eigval = eigval[order_idx]
    eigval = np.diag(eigval)
    eigvec = eigvec[:, order_idx]

    # (4) Get the matrix p where y = x @ p
    p = eigvec[:, :k]

    # (5) Get the explained variance
    explained_var = eigval[:k, :k]
    explained_var_ratio = explained_var.sum()/eigval.sum()

    # (6) Get the principal component
    comp = x @ p

    return comp, explained_var_ratio

if __name__ == "__main__":
    mat = np.array([[-1, 1, 0, 3],
                    [0, -1, 1, 4],
                    [11, 13, -2, 5]])
    result = pca(mat, 2)
