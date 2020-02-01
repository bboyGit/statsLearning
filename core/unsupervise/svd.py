import numpy as np
import pandas as pd

class svd:

    def __init__(self, mat, k=None):
        """
        :param mat: m*n matrix
        :param k: Int representing the number of singular values we'll keep. Default None, which means full svd.
        """
        self.mat = mat
        m, n = mat.shape
        if k == None or k == min(m, n):
            self.svd_type = 0
        elif k > min(m, n):
            raise Exception('k can not bigger than the number of rows and columns of mat')
        elif 0 < k < min(m, n):
            self.svd_type = 1
            self.k = k

    def decompose(self):
        """
        Desc: Do A = U @ M @ V
        """
        # (1) Compute eigenvalue and eigenvector of A.T @ A
        eigval, eigvec = np.linalg.eig(self.mat.T @ self.mat)
        eigval = pd.Series(eigval).sort_values(ascending=False)
        eigval[eigval.abs() < 10**(-8)] = 0
        r = len(eigval[eigval > 0])                   # rank of A.T @ A
        # (2) Format V and M
        M = np.diag(np.sqrt(eigval))
        V = eigvec[:, eigval.index]
        m, n = self.mat.shape
        if n >= m:
            M = M[:m, :]
        else:
            M = np.concatenate([M, np.zeros([m - n, n])], axis=0)
        # (3) Compute U
        eigval1, U = np.linalg.eig(self.mat @ self.mat.T)
        eigval1 = pd.Series(eigval1).sort_values(ascending=False)
        eigval1[eigval1.abs() < 10**(-8)] = 0
        U = U[:, eigval1.index]
        # (4) Cut U, M and V according to k and svd_type
        if self.svd_type == 0:
            return U, M, V
        elif self.svd_type == 1:
            return U[:, :self.k], M[:self.k, :self.k], V[:, :self.k]

if __name__ == "__main__":
    mat = np.array([[-1, 1, 0, 2],
                    [0, -1, 1, 1.3],
                    [1.2, 9, 0, 2.13]])
    print(svd(mat, k=2).decompose())
    print(svd(mat).decompose())