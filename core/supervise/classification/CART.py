
import numpy as np
import pandas as pd

class cart:
    """
    Desc: This is cart decision tree.
    """
    def __init__(self, data, response, tree_type):
        """
        :param data:
        :param response: str
        :param tree_type: str, regression or classification
        """
        self.data = data.copy()
        self.label = data[response].unique()
        self.response = response
        self.type = tree_type

    def gini(self, x):
        """
        Desc: Compute Gini index of a given series
        :param x: array
        :return: A float
        """
        uniq = np.unique(x)
        N = x.shape[0]
        prob = np.array([[x[x == i].shape[0]/N for i in uniq]])
        result = prob @ (1 - prob.T)
        return result[0, 0]

    def cond_gini(self, y, x):
        """
        Desc: Compute Gini index of y conditioning on x
        :param y: A series
        :param x: A series
        :return: A float
        """
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values

        xy = np.concatenate([x.reshape([len(x), 1]), y.reshape([len(y), 1])], axis=1)
        uniq = np.unique(x)
        N = len(x)
        prob_x = []
        Gini = []
        for i in uniq:
            xy_i = xy[x == i, :]
            n = xy_i.shape[0]
            prob_x.append(n/N)
            y_i = xy_i[:, -1]
            Gini.append(self.gini(y_i))
        prob_x = np.array([prob_x])
        Gini = np.array([Gini])
        result = (prob_x @ Gini.T)[0, 0]

        return result

    def mse(self, x):
        """
        Desc: Compute mean square error of a given series
        :param x: A sereis
        :return: A float
        """
        return np.sum((x - x.mean())**2)

    def cond_mse(self, y, x):
        """
        Desc: Compute mse(y|x = x1) + mse(y|x = x2)
        :return: A number
        """
        xy = np.concatenate([x.reshape([len(x), 1]), y.reshape([len(y), 1])], axis=1)
        uniq = np.unique(x)
        mse = []
        for i in uniq:
            xy_i = xy[x == i, :]
            y_i = xy_i[:, -1]
            mse_i = self.mse(y_i)
            mse.append(mse_i)
        mse = np.sum(mse)

        return mse

    def split_point(self, y, x, y_type, x_type):
        """
        Desc: Choose the best split point of of a given x(discrete, continuous)
        :param y: A series
        :param x: A series
        :param y_type: str, 'discrete' or 'continuous'
        :param x_type: str, 'discrete' or 'continuous'
        :return: A number(continuous x) or a str(discrete x)
        """
        if y_type == 'classification':
            # classification
            gini_idx = {}
            if x_type == 'discrete':
                uniq = x.unique()
                for i in uniq:
                    x_ = x.values.copy()
                    x_[x == i] = 0
                    x_[x != i] = 1
                    gini_idx[i] = self.cond_gini(y, x_)
            elif x_type == 'continuous':
                order_x = x.sort_values().values
                n = len(order_x) - 1
                T = np.round([(order_x[i] + order_x[i + 1]) / 2 for i in range(n)], 2)
                for i in T:
                    x_ = x.values.copy()
                    x_[x < i] = 0
                    x_[x >= i] = 1
                    gini_idx[i] = self.cond_gini(y, x_)

            gini_idx = pd.DataFrame(gini_idx, index=['gini_idx']).T
            optimal_point = gini_idx.idxmin()['gini_idx']
        else:
            # regression
            mse_idx = {}
            if x_type == 'discrete':
                uniq = x.unique()
                for i in uniq:
                    x_ = x.values.copy()
                    x_[x == i] = 0
                    x_[x != i] = 1
                    mse_idx[i] = self.cond_mse(y, x_)
            elif x_type == 'continuous':
                order_x = x.sort_values().values
                n = len(order_x) - 1
                T = np.round([(order_x[i] + order_x[i + 1]) / 2 for i in range(n)], 4)
                for i in T:
                    x_ = x.values.copy()
                    x_[x < i] = 0
                    x_[x >= i] = 1
                    mse_idx[i] = self.cond_mse(y, x_)
            mse_idx = pd.DataFrame(mse_idx, index=['mse_idx']).T
            optimal_point = mse_idx.idxmin()['mse_idx']

        return optimal_point

    def feature_select(self, Y, X):
        """
        Desc: choose the best feature in X
        :param Y: array
        :param X: A dataFrame
        :return: A tuple containing the best feature and split point and the corresponding loss value(gini or mse)
        """
        loss = {}
        point = {}
        for feature in X.columns:
            x_i = X[feature].copy()
            if isinstance(x_i.iloc[0], str):
                # Discrete independent variable
                point[feature] = self.split_point(Y, x_i, y_type=self.type, x_type='discrete')
                x_i = [0 if i == point[feature] else 1 for i in x_i.values]
            else:
                # Continuous independent variable
                point[feature] = self.split_point(Y, x_i, y_type=self.type, x_type='continuous')
                x_i = np.array([0 if i < point[feature] else 1 for i in x_i.values])
            loss[feature] = self.cond_gini(Y, x_i) if self.type == 'classification' else self.cond_mse(Y, x_i)

        loss = pd.DataFrame(loss, index=['loss']).T
        curr_node_name = loss.idxmin()['loss']
        loss_min = loss.min()['loss']
        optimal_point = point[curr_node_name]

        return curr_node_name, optimal_point, loss_min

    def split_df(self, data, curr_node_name, split_point):
        """
        Desc: Split the dataframe into 2 part by a specific feature and split point
        :param data: A DataFrame
        :param curr_node_name: A feature name
        :param split_point: A str or number
        :return: A tuple containing 2 dataframe
        """
        if isinstance(data[curr_node_name].iloc[0], str):
            # the best feature is discrete
            data0 = data.loc[data[curr_node_name] == split_point, :].copy()
            data1 = data.loc[data[curr_node_name] != split_point, :].copy()
            num1 = len(data1[curr_node_name].unique())
            data0.drop(curr_node_name, axis=1, inplace=True)
            if num1 == 1:
                data1.drop(curr_node_name, axis=1, inplace=True)
        else:
            # the best feature is continuous
            data0 = data.loc[data[curr_node_name] < split_point, :].copy()
            data1 = data.loc[data[curr_node_name] >= split_point, :].copy()

        return data0, data1

    def regress_tree(self, data, thresh):
        """
        Desc: method to generating a regression tree
        :param data: A dataframe representing the training set
        :param thresh: A float. The threshold of skip out recursion
        :return: A dict
        """
        data = data.copy()
        data.reset_index(drop=True, inplace=True)
        Y = data[[self.response]]
        X = data.drop(self.response, axis=1)
        num_feature = X.shape[1]
        obs = data.shape[0]
        if num_feature == 0:
            return Y.mean()[self.response]
        elif obs < 5*num_feature:
            return Y.mean()[self.response]
        else:
            curr_node_name, split_point, min_mse = self.feature_select(Y.values, X)
            if min_mse < thresh:
                return Y.mean()[self.response]
            else:
                result = {}
                data0, data1 = self.split_df(data, curr_node_name, split_point)
                result[(curr_node_name, 0, split_point)] = self.regress_tree(data0, thresh)
                result[(curr_node_name, 1, split_point)] = self.regress_tree(data1, thresh)

        return result

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
        uniq = np.unique(Y.values)
        K = len(uniq)
        if K == 1:
            return uniq[0]
        elif num_feature == 0:
            max_y = Y.reset_index().groupby(self.response).count().idxmax()['index']
            return max_y
        else:
            curr_node_name, split_point, min_gini = self.feature_select(Y.values, X)
            if min_gini < thresh:
                max_y = Y.reset_index().groupby(self.response).count().idxmax()['index']
                return max_y
            else:
                result = {}
                data0, data1 = self.split_df(data, curr_node_name, split_point)
                result[(curr_node_name, 0, split_point)] = self.class_tree(data0, thresh)
                result[(curr_node_name, 1, split_point)] = self.class_tree(data1, thresh)

        return result

    def generate(self, data, thresh):
        """
        Desc: method to generating a regression tree or classification tree
        :param data: A dataframe representing the training set
        :param thresh: A float. The threshold of skip out recursion
        :return: A dict
        """
        if self.type == 'regression':
            result = self.regress_tree(data, thresh)
        elif self.type == 'classification':
            result = self.class_tree(data, thresh)
        tree = {'root': result}
        return tree

    def print_tree(self, dt, tap):
        for i, j in dt.items():
            print(tap, i)
            if isinstance(j, dict):
                self.print_tree(j, tap+'   ')
            else:
                if self.type == 'classification':
                    print(tap+'   ', ' Class: ', j)
                elif self.type == 'regression':
                    print(tap+'  ', ' Value: ', j)


if __name__ == "__main__":
    import sklearn.datasets as dataset
    # Classification tree
    breast_cancer = dataset.load_breast_cancer()
    train_x = breast_cancer['data'][:500, :3]
    train_x = pd.DataFrame(train_x, columns=breast_cancer.feature_names[:3])
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    train_y = pd.DataFrame(train_y, columns=['cancer'])
    train = pd.concat([train_y, train_x], axis=1)

    cls_tree = cart(train, 'cancer', tree_type='classification')
    cart1 = cls_tree.generate(train, 0.001)
    cls_tree.print_tree(cart1, '   ')

    # Regression tree
    boston = dataset.load_boston()
    y = boston['target'][:100].reshape(100, 1)
    x = boston['data'][:100, :3].reshape(100, 3)
    y = pd.DataFrame(y, columns=['price'])
    x = pd.DataFrame(x, columns=boston.feature_names[:3])
    xy = pd.concat([y, x], axis=1)
    reg_tree = cart(xy, 'price', tree_type='regression')
    cart2 = reg_tree.generate(xy, 1)
    reg_tree.print_tree(cart2, '   ')


