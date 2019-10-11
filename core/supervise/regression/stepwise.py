
import numpy as np
import pandas as pd
from core.supervise.regression.ols import ols
from core.supervise.regression.fitness import r_square, aic, bic

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
    best_num = np.array(M).argmin()
    idx_in = np.array(idx_in)
    select_x = X[:, idx_in[:best_num]]
    best_model = ols(select_x, Y, const=True)

    result = {'idx_in': idx_in, 'select_x': select_x, 'model': best_model}

    return result

def backward(Y, X, func):


    return

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

