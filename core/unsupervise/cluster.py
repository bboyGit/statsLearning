import numpy as np

class cluster:

    def minkowski_distance(self, x, y, p):
        """
        Desc: this method allows us to compute minkowski distance
        :param p: Int, representing the power of minkowski distance
        :return: Float
        """
        if p == 1:
            dist = np.sum(np.abs(x - y))              # Manhattan distance
        elif p == 2:
            dist = np.sum((x - y)**2).sqrt()          # Euclidean distance
        elif p == np.inf:
            dist = np.max(np.abs(x - y))              # Chebyshev distance
        else:
            dist = np.sum((x - y)**p)**(1/p)
        return dist

    def Mahalanobis_distance(self, i, j, data):
        cov_inv = np.linalg.inv(np.cov(data.T))
        x = data[[i], :]
        y = data[[j], :]
        dist = np.sqrt((x - y) @ cov_inv @ (x - y).T)[0, 0]
        return dist

    def dist_between(self, group1, group2, linkage_type=0, p=2):
        """
        Desc: Compute distance between 2 clusters: group1 and group2
        :param linkage_type: 0, 1 or 2 representing different kinds of linkage way of 2 clusters.
        :param p: The power of minkowski distance. Default 2 which means Euclidean distance.
        :return: A float
        """
        n1, d1 = group1.shape
        n2, d2 = group2.shape
        if linkage_type in (0, 1, 2):
            dist_mat = np.zeros(shape=[n1, n2])
            for i in range(n1):
                for j in range(n2):
                    x = group1[i, :].reshape([d1, 1])
                    y = group2[j, :].reshape([d2, 1])
                    dist_mat[i, j] = self.minkowski_distance(x, y, p)
            if linkage_type == 0:            # single linkage
                dist = dist_mat.min()
            elif linkage_type == 1:          # complete linkage
                dist = dist_mat.max()
            elif linkage_type == 2:          # average linkage
                dist = dist_mat.mean()
        elif linkage_type == 3:              # center linkage
            center1 = group1.mean(axis=0)
            center2 = group2.mean(axis=0)
            dist = self.minkowski_distance(center1, center2, p)
        else:
            raise Exception('linkage_type can only be 0, 1 or 2')
        return dist

class hierarchy_cluster(cluster):
    """
    Desc: Execute hierarchical cluster by bottom-up way.
    """
    def __init__(self, data, linkage_type=0, p=2):
        """
        :param data: DataFrame or array of samples
        :param linkage_type: 0,1,2 or 3. representing different way of measureing distance of 2 groups.
        :param p: Int. The power of minkowski distance
        """
        self.data = data
        self.linkage_type = linkage_type
        self.p = p

    def fit(self):
        n, m = self.data
        groups = [[self.data[[i], :] for i in range(n)]]
        num_cluster = n
        count = 0
        while num_cluster > 1:
            # (1) Get the info of current clusters
            pre_group = groups[count].copy()
            dist_cluster = np.zeros(shape=[num_cluster, num_cluster])
            # (2) Compute distances between each groups
            for i in range(num_cluster - 1):
                for j in range(i+1, num_cluster):
                    group1 = pre_group[i]
                    group2 = pre_group[j]
                    dist_cluster[i, j] = self.dist_between(group1, group2, self.linkage_type, self.p)
            # (2) Combine the closet 2 groups into 1
            min_dist = dist_cluster[dist_cluster > 0].min()
            combine_id = np.where(dist_cluster == min_dist)
            id1 = combine_id[0][0]
            id2 = combine_id[1][0]
            combine_group = np.concatenate([pre_group[id1], pre_group[id2]], axis=1)
            del pre_group[id1], pre_group[id2]
            # (3) Update groups
            updated_group = pre_group.append(combine_group)
            num_cluster = len(updated_group)
            count += 1
            groups.append(updated_group)

        return groups

class partition_cluster(cluster):
    """
    Desc: Execute partition cluster
    """
    def __init__(self, data, p, k, method, max_iter: int, thresh: float):
        """
        :param data: DataFrame or array of samples
        :param p: Int. The power of minkowski distance
        :param k: Int. The number of clusters
        :param method: A int representing the method of measuring center. {0: means, 1:median}
        :param max_iter: A int representing the max iteration times
        :param thresh: A float representing the threshold of breaking loop
        """
        self.data = data
        self.p = p
        self.k = k
        self.method = method
        self.max_iter = max_iter
        self.thresh = thresh
        n = len(data)
        self.center = data[[np.random.randint(0, n-1, k)], :]           # The initial center of partition cluster

    def fit(self):
        n, m = self.data.shape
        count = 0
        dist2center = np.zeros(shape=[n, self.k])
        while count <= self.max_iter:
            # (1) Compute each distance between each samples and each centers
            for i in range(n):
                for k in range(self.k):
                    dist2center[i, k] = self.minkowski_distance(self.data[i, :], self.center[k, :], p=self.p)
            # (2) Each samples belong to the cluster whose center and the sample are closet
            cluster_idx = dist2center.argmin(axis=1)
            # (3) Update centers
            update_center = self.center.copy()
            for k in range(self.k):
                kth_group = self.data[cluster_idx == k, :]
                if self.method == 0:
                    update_center[k, :] = kth_group.mean(axis=0)
                elif self.method == 1:
                    update_center[k, :] = kth_group.median(axis=0)
                else:
                    raise Exception('method can only be 0 or 1')
            # (4) Judge wither we will break the loop
            diff = np.mean(np.abs(update_center - self.center))
            self.center = update_center
            if diff < self.thresh:
                break
            count += 1

        return cluster_idx


