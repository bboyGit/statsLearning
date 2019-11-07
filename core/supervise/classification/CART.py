
import numpy as np
import pandas as pd

class cart:
    """
    Desc: This is cart decision tree.
    """
    def __init__(self, data, response):
        self.data = data.copy()
        self.label_name = response
        self.label = data[response].unique()
        self.response = response

    def gini(self, x):
        """
        Desc: Compute Gini index of a given series
        :param x: A Series
        :return: A float
        """
        uniq = x.unique()
        N = x.shape[0]
        prob = np.array([[x[x == i].shape[0]/N for i in uniq]])
        result = prob @ (1 - prob.T)
        return result

    def cond_gini(self, y, x):
        """
        Desc: Compute Gini index of y conditioning on x
        :param y: A series
        :param x: A series
        :return: A float
        """

        return

    def mse(self, x):
        """
        Desc: Compute mean square error of a given series
        :param x: A sereis
        :return: A float
        """
        return np.sum((x - x.mean())**2)

    def cond_mse(self,y, x):
        pass

    def split_point(self):
        pass

    def feature_select(self, Y, X):
        pass

    def regress_tree(self, data, thresh):
        """
        Desc: method to generating a regression tree
        :param data: A dataframe representing the training set
        :param thresh: A float. The threshold of skip out recursion
        :return: A dict
        """
        return

    def class_tree(self, data, thresh):
        """
        Desc: method to generating a classification tree
        :param data: A dataframe representing the training set
        :param thresh: A float. The threshold of skip out recursion
        :return: A dict
        """
        data = data.copy()
        data.reset_index(drop=True, inplace=True)
        Y = data[[self.response]]
        X = data.drop(self.response, axis=1)
        num_feature = X.shape[1]
        uniq = Y.unique()
        K = len(uniq)
        if K == 1:
            return uniq[0]
        elif num_feature == 0:
            max_y = Y.reset_index().groupby(self.response).count().idxmax()['index']
            return max_y
        else:
            feature, curr_node_name, min_gini = self.feature_select(Y, X)
            if min_gini < thresh:
                max_y = Y.reset_index().groupby(self.response).count().idxmax()['index']
                return max_y
            else:
                result = {}
                for val in feature:
                    data_i = data.loc[data[curr_node_name] == val, :].copy()
                    num = len(data_i[curr_node_name].unique())
                    if num == 1:
                        data_i.drop(curr_node_name, axis=1, inplace=True)
                    result[(curr_node_name, val)] = self.class_tree(data_i, thresh)

        return result

    def generate(self, data, thresh, kind):
        """
        Desc: method to generating a regression tree or classification tree
        :param data: A dataframe representing the training set
        :param thresh: A float. The threshold of skip out recursion
        :param kind: A str indicating what kind of tree we want to create
        :return: A dict
        """
        if kind == 'regression':
            result = self.regress_tree(data, thresh)
        elif kind == 'classification':
            result = self.class_tree(data, thresh)
        tree = {'root': result}
        return tree

    def print_tree(self):
        pass


if __name__ == "__main__":
    import sklearn.datasets as dataset

    breast_cancer = dataset.load_breast_cancer()
    train_x = breast_cancer['data'][:500, :]
    train_x = pd.DataFrame(train_x, columns=breast_cancer.feature_names[:])
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    train_y = pd.DataFrame(train_y, columns=['cancer'])
    train = pd.concat([train_y, train_x], axis=1)