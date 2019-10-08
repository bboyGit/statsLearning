
import pandas as pd
import numpy as np

def errorRate(y, hat_y):
    """
    Desc: Compute error rate of a fitted value to real value
    Parameters:
      y: An array or Series indicating fitted value
      hat_y: An array or Series indicating fitted value
    Return: A float indicate error rate.
    """
    confuseMat = pd.crosstab(y, hat_y).values
    right_num = confuseMat.diagonal().sum()
    tot_num = confuseMat.sum()
    right_rate = right_num/tot_num
    error_rate = 1 - right_rate

    return error_rate

def mse(y, hat_y):
    """
    Desc: Compute mean square error of a fitted value to real value
    Parameters:
      y: An array or Series indicating fitted value
      hat_y: An array or Series indicating fitted value
    Return: A float indicate error rate.
    """
    diff = (y - hat_y)**2
    n = len(y)
    result = diff/n
    return result

def cv(data, m, func, kind, **kwargs):
    """
    Desc: Do k-fold Cross Validation
    Parameters:
      data: A dataframe or matrix containing train set whose last column represent y and the remain columns are x.
      m: An integer indicating the number of fold.
      func: A function or a string of a function name
      kind: A str. regression or classification.
      kwargs: Additional keyword argument
    Return: An float indicating CV.
    Note: If k == 1 then it is leave one out cross validation.
    Algorithm Schedule:
        (1) Divide the data set into k groups and fit statistical learning model for k times.
        (2) Each times we use k-1 groups of data to fit model and calculate test error rate or mse by the remain 1 group.
        (3) Calculate the mean of test error rate or mse of those k time calculation.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if isinstance(func, str):
        func = eval(func)
    # (1) Set the k split points of data
    n = data.shape[0] - 1
    split_point = np.linspace(0, n, m + 1).round()
    split_point = [int(i) for i in split_point]

    # (2) Each times we use k-1 groups of data to fit model and calculate test error rate or mse by the remain 1 group.
    test_error = []

    for i in range(m):
        start_idx = split_point[i]
        end_idx = split_point[i + 1]
        if end_idx == n:
            validation = data.iloc[start_idx:, :]
            train = data.iloc[: start_idx, :]
        else:
            validation = data.iloc[start_idx: end_idx, :]
            if start_idx == 0:
                train = data.iloc[end_idx:, :]
            else:
                train = pd.concat([data.iloc[:start_idx, :], data.iloc[end_idx:, :]], axis=0)
        train_x = train.iloc[:, :-1]
        train_y = train.iloc[:, [-1]]
        validation_x = validation.iloc[:, :-1]
        validation_y = validation.iloc[:, [-1]]
        hat = func(train_x, train_y, validation_x, **kwargs)
        if kind == 'regression':
            error = mse(y=validation_y, hat_y=hat)
        elif kind == 'classification':
            error = errorRate(y=validation_y, hat_y=hat)
        test_error.append(error)

    # (3) Calculate the mean of test error rate or mse of those k time calculation
    test_error = np.array(test_error)
    result = test_error.mean()

    return result


if __name__ == "__main__":
    import sklearn.datasets as dataset
    from core.supervise.classification.knn import knn
    breast_cancer = dataset.load_breast_cancer()
    train = breast_cancer['data'][:500, :]
    cv(train, 10, 'knn', k=10)                        # 有 Bug -------------------- - -



