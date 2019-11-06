
import numpy as np
import pandas as pd

class id3_c45:
    """
    Desc: Decision tree object, which is used to create ID3 or C4.5 decision tree(a non-parametric supervised method).
    """
    def __init__(self, data, response):
        self.data = data.copy()
        self.label_name = response
        self.label = data[response].unique()

    def entropy(self, x):
        """
        Desc: Compute empirical entropy of a given discrete variable x
        :param x: A series or array
        :return: A float indicating its empirical entropy
        """
        # (1) Deal with input data type
        if not isinstance(x, np.ndarray) and not isinstance(x, pd.Series):
            raise TypeError('x must be an array or a series')
        if isinstance(x, pd.Series):
            x = x.values

        # (2) Compute entropy
        uniq = np.unique(x)
        N = len(x)
        prob = []
        log_p = []
        for i in uniq:
            x_i = x[x == i]
            n = len(x_i)
            prob.append(n/N)
            log_p.append(np.log2(n/N))
        prob = np.array([prob])
        log_p = np.array([log_p])
        ent = -prob @ log_p.T
        result = ent[0, 0]

        return result

    def cond_entropy(self, y, x):
        """
        Desc: Compute entropy of y conditioning on x. H(Y|X)
        :param y: A series or array
        :param x: A series or array whose length is equal to y
        :return: A float indicating conditional entropy.
        """
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values

        # (2) Compute conditional entropy
        xy = np.concatenate([x.reshape([len(x), 1]), y.reshape([len(y), 1])], axis=1)
        uniq = np.unique(x)
        N = len(x)
        prob_x = []
        ent = []
        for i in uniq:
            xy_i = xy[x == i, :]
            n = xy_i.shape[0]
            prob_x.append(n/N)
            y_i = xy_i[:, -1]
            ent_i = self.entropy(y_i)
            ent.append(ent_i)
        prob_x = np.array([prob_x])
        ent = np.array([ent])
        cond_ent = prob_x @ ent.T
        cond_ent = cond_ent[0, 0]

        return cond_ent

    def mutual_info(self, y, x):
        """
        Desc: Compute mutual infomation(information gain). g(y, x) = H(y) - H(y|x)
        :param y: A series or array
        :param x: A series or array whose length is equal to y
        :return: A float indicating mutual info
        """
        ent_y = self.entropy(y)
        cond_ent = self.cond_entropy(y, x)
        info_gain = ent_y - cond_ent

        return info_gain

    def mutual_info_ratio(self, y, x):
        """
        Desc: Compute information gain ratio. g(y, x)/H(x)
        :param y: A series or array
        :param x: A series or array whose length is equal to y
        :return: A float indicating mutual info ratio
        """
        info_gain = self.mutual_info(y, x)
        ent_x = self.entropy(x)
        info_gain_ratio = info_gain/ent_x

        return info_gain_ratio

    def split_point(self, y, x):
        """
        Desc: This method is used to deal with continuous variable
        :param y: A series or array
        :param x: A series or array whose length is equal to y
        :return: A float indicating split point of x
        """
        order_x = x.sort_values().values
        n = len(order_x) - 1
        T = np.round([(order_x[i] + order_x[i + 1])/2 for i in range(n)], 2)
        info_gain = {i: None for i in T}
        for point in T:
            x_ = x.values.copy()
            x_[x_ < point] = 0
            x_[x_ >= point] = 1
            info_gain[point] = self.mutual_info(y, x_)
        info_gain = pd.DataFrame(info_gain, index=['info_gain']).T
        optimal_point = info_gain.idxmax()['info_gain']
        x = ['< {}'.format(optimal_point) if i < optimal_point else ">= {}".format(optimal_point) for i in x.values]
        x = pd.Series(x)

        return x

    def feature_select(self, Y, X, fun):
        """
        Desc: It's a feature selection method used to select feature in X by criterion fun.
        :param Y: A series
        :param X: A dataframe
        :param fun: A function object, self.mutual_info or self.mutual_info_ratio
        :return: A tuple
        """
        info_gain = {}
        x = {}
        for feature in X.columns:
            x_i = X[feature]
            if isinstance(x_i.iloc[0], str):
                # Discrete variable
                info_gain[feature] = fun(Y.values, x_i)
            else:
                # Deal with continuous variable
                x_i = self.split_point(Y.values, x_i)
                info_gain[feature] = fun(Y.values, x_i)
            x[feature] = x_i

        info_gain = pd.DataFrame(info_gain, index=['info_gain']).T
        curr_node_name = info_gain.idxmax()['info_gain']
        X[curr_node_name] = x[curr_node_name].copy()
        info_gain_max = info_gain.max()['info_gain']

        return X, curr_node_name, info_gain_max

    def generate(self, data, thresh, response, method):
        """
        Desc: Generate id3 decision tree recursively
        :param data: A dataframe containing features and response
        :param thresh: A float indicating minimum threshold of information gain
        :param response: A str representing name of response
        :param method: A str, ID3 or C45.
        :return: A dict indicating the framework of tree.
        """
        # (1) Determine classification method
        if method == 'id3':
            fun = self.mutual_info
        elif method == 'c45':
            fun = self.mutual_info_ratio
        # (2) Recursively generate tree
        data = data.copy()
        data.reset_index(drop=True, inplace=True)
        Y = data[[response]]
        X = data.drop(response, axis=1)
        n = X.shape[1] if isinstance(X, pd.DataFrame) else 1
        uniq = np.unique(Y.values)
        K = len(uniq)

        if K == 1:
            return uniq[0]
        elif n == 0:
            max_y = Y.reset_index().groupby(response).count().idxmax()['index']
            return max_y
        else:
            X, curr_node_name, info_gain_max = self.feature_select(Y, X, fun)
            if info_gain_max < thresh:
                max_y = Y.reset_index().groupby(response).count().idxmax()['index']
                return max_y
            else:
                feature = X.loc[:, curr_node_name].unique()
                result = {}
                data_ = pd.concat([Y, X], axis=1)
                for val in feature:
                    data_i = data_.loc[data_[curr_node_name] == val, :].copy()
                    data_i.drop(curr_node_name, axis=True, inplace=True)
                    ent_i = self.entropy(data_i[response])
                    result[(curr_node_name, val)] = ent_i, self.generate(data_i, thresh, response, method)
                return result

    def generate_tree(self, data, thresh, response, method):
        dt1 = {}
        dt = self.generate(data, thresh, response, method)
        ent = self.entropy(data[response])
        dt1['root'] = ent, dt
        return dt1

    def print_tree(self, dt, tap='   '):
        """
        Desc: print a decision tree who's generated by self.generate
        :param dt: A dict representing the framework of a decision tree
        :return: None
        """
        for i, j in dt.items():
            print(tap, i)
            if isinstance(j[-1], dict):
                self.print_tree(j[-1], tap+"   ")
            else:
                print(tap + '   ', 'Class: ', j[-1], ' ent: ', round(j[0], 4))
        return

if __name__ == "__main__":

    from sklearn import tree
    import sklearn.datasets as dataset

    breast_cancer = dataset.load_breast_cancer()
    train_x = breast_cancer['data'][:500, :]
    train_x = pd.DataFrame(train_x, columns=breast_cancer.feature_names[:])
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    train_y = pd.DataFrame(train_y, columns=['cancer'])
    train = pd.concat([train_y, train_x], axis=1)
    dt = id3_c45(train, response='cancer')
    id3 = dt.generate_tree(train, thresh=0.1, response='cancer', method='id3')
    dt.print_tree(id3, ' ')
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # fit = clf.fit(train_x, train_y)
    # tree.plot_tree(fit)












