import numpy as np
from copy import deepcopy
from numpy.linalg import det, inv

class GMM:

    def __init__(self, init_mean, init_cov, init_p, max_iter: int, x, thresh: float):
        """
        :param init_mean: matrix with m rows and K columns. Initial guess of means.
        :param init_cov: List of covariance matrix(m*m). Initial guess of covariance matrix.
        :param init_p: array with 1 column and K rows. Initial guess of state probability vector.
        :param max_iter: int. maximum iteration steps.
        :param x: matrix with N(number of observation) rows and m(number of feature) columns. Observation sequence.
        :param thresh: float. Threshold of difference of Q
        """
        self.max_iter = max_iter
        self.K = len(init_cov)
        self.x = x
        self.init_mean = init_mean
        self.init_p = init_p
        self.init_cov = init_cov
        self.thresh = thresh

    def __calcu_r_ik(self, x, p, u, cov_lst):
        """
        Desc: Calculate p(z|x, theta_t)
        :param x:  matrix with N rows and m columns. Observation sequence.
        :param p: array with 1 column and K rows. State probability vector.
        :param u: matrix with m rows and K columns. Means.
        :param cov_lst: List of covariance matrix(m*m). Covariance matrix.
        :return: A matrix with N rows and K columns
        """
        pdf = lambda x_i, u_k, cov_k, m: (2*np.pi)**(-m/2) * det(cov_k)**(-1/2) * np.exp(-0.5*(x_i - u_k).T @ inv(cov_k) @ (x_i - u_k))[0]
        r = []
        N, m = x.shape
        for i in range(N):
            r_i = np.array([pdf(x[[i], :].T, u[:, [k]], cov_lst[k], m) * p[k, :] for k in range(self.K)]).T
            r.append(r_i/r_i.sum())
        r = np.concatenate(r, axis=0)
        return r

    def fit(self):
        # (1) copy those 3 model parameters, relative materials and observations
        p = self.init_p.copy()
        u = self.init_mean.copy()
        cov_lst = deepcopy(self.init_cov)
        det_lst = np.array([det(i) for i in cov_lst])
        inv_lst = [inv(i) for i in cov_lst]
        x = self.x.copy()
        N, m = x.shape
        # (2) execute EM algorithm to get estimate of p, u and cov
        def q2(r, x, u, inv_lst):
            Q2 = 0
            for i in range(N):
                for j in range(self.K):
                    Q2 -= 0.5 * r[i, j] * (x[[i], :] - u[:, [j]].T) @ inv_lst[j] @ (x[[i], :] - u[:, [j]].T).T
            return Q2
        count = 0
        while count <= self.max_iter:
            print(count)
            # (2.1) E step: estimate Q function <=> estimate P(z_i|y_i, theta_t)
            r = self.__calcu_r_ik(x, p, u, cov_lst)
            Q1 = np.sum(r * np.log(p).ravel()) - 0.5 * np.sum(r * np.log(det_lst))
            # Q2 = - 0.5 * np.sum([r[:, [k]] * (x - u[:, k]) @ inv_lst[k] @ (x - u[:, k]).T for k in range(self.K)])
            Q2 = q2(r, x, u, inv_lst)
            Q = Q1 + Q2
            # (2.2) M step: maximize Q function by update p, u and cov_lst
            # (2.2.1) update p
            p = np.array([r[:, k].sum()/N for k in range(self.K)]).reshape(self.K, 1)
            # (2.2.2) update u
            u = [((x * r[:, [k]]).sum(axis=0)/r[:, k].sum()).reshape(m, 1) for k in range(self.K)]
            u = np.concatenate(u, axis=1)
            # (2.2.3) update cov
            cov_lst = [(r[:, k] * (x - u[:, k]).T @ (x - u[:, k]))/r[:, k].sum() for k in range(self.K)]
            det_lst = np.array([det(i) for i in cov_lst])
            inv_lst = [inv(i) for i in cov_lst]
            # (2.3) judge whether we can break the loop
            Q1_update = np.sum(r * np.log(p).ravel()) - 0.5 * np.sum(r * np.log(det_lst))
            # Q2_update = - 0.5 * np.sum([r[:, [k]] * (x - u[:, k]) @ inv_lst[k] @ (x - u[:, k]).T for k in range(self.K)])
            Q2_update = q2(r, x, u, inv_lst)
            Q_update = Q1_update + Q2_update
            if np.abs(Q - Q_update) < self.thresh:
                r = self.__calcu_r_ik(x, p, u, cov_lst)
                break
            count += 1
        # (3) tidy result
        result = {'p': p, 'u': u, 'cov': cov_lst, 'p(z|x, theta)': r}

        return result


if __name__ == "__main__":
    # 随机生成1000个样本，每个样本有0.7的概率处于第一个高斯分布、0.3的概率处于第二个高斯分布。
    data = np.empty(shape=(1000, 2))
    for i in range(len(data)):
        p = np.random.rand(1)
        if p > 0.3:
            data[i, :] = np.random.multivariate_normal(mean=np.array([1, 1.2]), cov=np.array([[1, 0.9], [0.9, 1.6]]))
        else:
            data[i, :] = np.random.multivariate_normal(mean=np.array([0.7, 0.2]), cov=np.array([[1.4, 1], [1, 0.8]]))

    init_u = np.array([[0.5, 0.5], [1, 0.4]])
    init_cov = [np.array([[2, 0.7], [0.7, 1.9]]), np.array([[3, 1.2], [1.2, 2.1]])]
    init_p = np.array([[0.2, 0.8]]).T
    gmm = GMM(init_u, init_cov, init_p, max_iter=500, x=data, thresh=10**(-8))
    result = gmm.fit()
    print(result)