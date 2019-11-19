
import numpy as np
import pandas as pd

def normalize(x, method):
    """
    Desc: normalize features of any model
    :param x: A dataFrame or 2-D array
    :param method: str indicating the method to normalize x
    :return: dataframe or 2-D array
    """
    x = x.copy()
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if method == 'max_min':
        result = x.apply(lambda w: (w - w.min())/(w.max() - w.min()))
    elif method == 'normal':
        result = x.apply(lambda w: (w - w.mean())/w.std())
    return result

if __name__ == '__main__':
    X = np.array([[3, 1, 1.3, 0.6],
                  [4.5, 1.1, 1, 0.49],
                  [1.9, 0.89, 0.87, 0.5],
                  [8, 1.7, 1, 0.6],
                  [5.6, 0.77, 0.4, 0.77]])
    normalize(X, method='max_min')

