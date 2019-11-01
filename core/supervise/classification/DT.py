
import numpy as np
import pandas as pd

class DT:
    """
    Desc: It is a decision tree object, which is used to create and trim tree.
    """
    bf = pd.DataFrame({'agree': [0, 0, 1, 0, 0, 1, 1, 1],
                       'height': ['low', 'med', 'high', 'high', 'low', 'med', 'med', 'high'],
                       'weight': ['thin', 'fat', 'med', 'med', 'thin', 'fat', 'thin', 'med'],
                       'age': [20, 40, 30, 35, 25, 35, 25, 30]})

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
        h = -prob @ log_p.T
        result = h[0, 0]

        return result
    
    def cond_entropy(self, y, x):
        """
        Desc: Compute entropy of y conditioning on x. H(Y|X)
        :param y: A series or array
        :param x: A series or array whose length is equal to y
        :return: A float indicating conditional entropy.
        """

        return