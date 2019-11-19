
import numpy as np
import pandas as pd

def add_const(x):
    N = x.shape[0]
    one = np.ones([N, 1])
    if isinstance(x, np.ndarray):
        result = np.concatenate((one, x), axis=1)
    elif isinstance(x, pd.DataFrame):
        x = x.values
        result = np.concatenate((one, x), axis=1)

    return result