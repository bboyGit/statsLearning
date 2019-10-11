
import numpy as np
import pandas as pd
from core.supervise.regression.ols import ols
from core.supervise.regression.fitness import r_square, aic, bic
from inspect import signature

def forward(Y, X, func):
    """
    Desc: Execute the forward stepwise regression. Start with zero variable.
    Return: A dict contain the selected X and the corresponding fitted model
    """
    n, p = X.shape
    idx_in = []               # record the idx of variable who already been chosen.
    idx_out = list(range(p))  # record the idx of variable who is not  been chosen yet.
    M = []

    # (1) Iterate all the possible number of x
    max_possible_num = min(n, p)
    for num in range(max_possible_num):
        Rss = {i: None for i in idx_out}

        # (2) Iterate all variables who's not been chosen yet
        for i in idx_out:
            x = X[:, [i]]
            if len(idx_in) != 0:
                x_already_in = X[:, idx_in]
                x = np.concatenate([x, x_already_in], axis=1)
            model = ols(x, Y, const=True)
            resid = model['resid']
            Rss[i] = np.sum(resid**2)

        # (3) Let the variable with minimum Rss in
        Rss = pd.DataFrame(Rss, index=['rss']).T
        new_idx_in = Rss.idxmin()['rss']
        idx_in.append(new_idx_in)
        idx_out.remove(new_idx_in)

        # (4) Calculate aic, bic or adjust r square of the current model Y ~ X[:, idx_in]
        x_already_in = X[:, idx_in]
        Model = ols(x_already_in, Y, const=True)
        Y_hat = Model['fit_value']
        d = len(idx_in)
        criterion = func(Y, Y_hat, d)
        M.append(criterion)

    # (5) Get the optimal model from M
    if 'adjust' in signature(func).parameters:
        best_num = np.array(M).argmax() + 1
    else:
        best_num = np.array(M).argmin() + 1
    idx_in = np.array(idx_in)
    select_x = X[:, idx_in[:best_num]]
    best_model = ols(select_x, Y, const=True)

    result = {'idx_in': idx_in, 'select_x': select_x, 'model': best_model}

    return result

def backward(Y, X, func):
    """
    Desc: Execute the forward stepwise regression. Start with zero variable.
    """
    n, p = X.shape
    idx_in = list(range(p))
    idx_out = []
    M = []

    # (1) Iterate all the possible number of x
    Model = ols(X, Y, const=True)
    Y_hat = Model['fit_value']
    d = len(idx_in)
    criterion = func(Y, Y_hat, d)
    M.append(criterion)

    for num in range(p - 1):
        Rss = {i: None for i in idx_in}

        # (2) Iterate all variables who's already been chosen
        for i in idx_in:
            idx_in1 = idx_in.copy()
            idx_in1.remove(i)
            x = X[:, idx_in1]
            model = ols(x, Y, const=True)
            resid = model['resid']
            Rss[i] = np.sum(resid ** 2)

        # (3) Let the variable with minimum Rss still in idx_in
        Rss = pd.DataFrame(Rss, index=['rss']).T
        drop_idx = Rss.idxmin()['rss']
        idx_in.remove(drop_idx)
        idx_out.append(drop_idx)

        # (4) Calculate aic, bic or adjust r square of the current model Y ~ X[:, idx_in]
        x_still_in = X[:, idx_in]
        Model = ols(x_still_in, Y, const=True)
        Y_hat = Model['fit_value']
        d = len(idx_in)
        criterion = func(Y, Y_hat, d)
        M.append(criterion)

        # (5) Get the optimal model from M
        if 'adjust' in signature(func).parameters:
            out_num = np.array(M).argmax()
        else:
            out_num = np.array(M).argmin()
        idx_out = np.array(idx_out)
        select_x = X[:, idx_out[out_num:]]
        best_model = ols(select_x, Y, const=True)

        result = {'idx_out': idx_out, 'select_x': select_x, 'model': best_model}

    return result

def step(Y, X, direction='forward', criterion='aic'):
    """
    Desc: Do stepwise regression
    Parameters:
      Y: A column vector contain dependent variable
      X: A matrix contain explanatory variables
      direction: A str between 'forward' or 'backward'
      criterion: A str among 'adj_r2', 'aic' and 'bic'
    Return: A dict contain the selected X and the corresponding fitted model
    """
    if criterion == 'aic':
        func = aic
    elif criterion == 'bic':
        func = bic
    else:
        func = r_square

    if direction == 'forward':
        result = forward(Y, X, func)
    else:
        result = backward(Y, X, func)

    return result


if __name__ == "__main__":
    import sklearn.datasets
    boston = sklearn.datasets.load_boston()
    y = boston['target']
    y = y.reshape(y.shape[0], 1)
    x = boston['data']
    regre = step(y, x, 'forward', 'adj_r2')
