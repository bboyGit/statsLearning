import numpy as np

class distance:

    def __init__(self, x, y, X=None):
        self.x = x.reshape([len(x), 1])
        self.y = y.reshape([len(y), 1])
        self.X = X                # x and y should be a row of X

    def minkowski_distance(self, p):
        """
        Desc: this method allows us to compute minkowski distance
        :param p: Int, representing the power of minkowski distance
        :return: Float
        """
        if p == 1:
            dist = np.sum(np.abs(self.x - self.y))              # Manhattan distance
        elif p == 2:
            dist = np.sum((self.x - self.y)**2).sqrt()          # Euclidean distance
        elif p == np.inf:
            dist = np.max(np.abs(self.x - self.y))              # Chebyshev distance
        else:
            dist = np.sum((self.x - self.y)**p)**(1/p)
        return dist

    def Mahalanobis_distance(self):
        cov_inv = np.linalg.inv(np.cov(self.X.T))
        dist = np.sqrt((self.x - self.y).T @ cov_inv @ (self.x - self.y))[0, 0]
        return dist

if __name__ == "__main__":
    data = np.random.normal(1, 1, size=[10, 5])
    d = distance(data[1, :], data[2, :], data)
    print(d.minkowski_distance(p=1))
    print(d.Mahalanobis_distance())