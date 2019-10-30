import numpy as np
import pandas as pd

def naiveBayes(trainX, trainY, testX, lam):
    """
    Desc: Classify test set by train set in naive bayes classifier.
    :param trainX: An array or a DataFrame which contains features of train set. Each row represent an observation.
    :param trainY: An array or a DataFrame which contains response of train set. Each row is an observation.
    :param testX: An array or a DataFrame which contains features of test set. Each row represent an observation.
    :param lam: A positive number. When it's zero, execute MLE, otherwise execute bayes estimation.
    :return: An array with 1 columns which is the class corresponding to that row of testX.
    Note:
        Assumption of naive bayes:
        1. Features are conditionally independent and subject to normal distribution, which means
        (x1, x2, ... xp|Y) ~ N(u, cov) while cov is a diagonal matrix.
        In essence, naiveBayes with gaussian distribution assumption is a QDA classifier who has
        conditional independent assumption.
    """
    if isinstance(trainX, np.ndarray):
        trainX = pd.DataFrame(trainX)
    if isinstance(trainY, np.ndarray):
        trainY = pd.DataFrame(trainY, columns=['y'])
    else:
        trainY.columns = ['y']
    if isinstance(testX, np.ndarray):
        testX = pd.DataFrame(testX)
    # (1) Divide train set into groups
    train = pd.concat([trainX, trainY], axis=1)
    category = trainY.iloc[:, 0].unique()
    N = train.shape[0]
    K = len(category)
    p = trainX.shape[1]

    # (2) Calculate prior probability, mean and cov of X of each class
    prior = {i: None for i in category}
    u = {i: None for i in category}
    cov = {i: None for i in category}
    cov_inv, cov_det = cov.copy(), cov.copy()
    for i in category:
        sub_train = train.query('y == @i').copy()
        sub_trainx = sub_train.iloc[:, :-1]
        prior[i] = (sub_train.shape[0] + lam) / (train.shape[0] + K * lam)
        ui = sub_trainx.iloc[:-1].mean().to_numpy()
        u[i] = ui.reshape(p, 1)
        cov_i = sub_trainx.cov()
        cov[i] = np.diag(np.diag(cov_i))
        cov_inv[i] = np.linalg.inv(cov[i])
        cov_det[i] = np.linalg.det(cov[i])

    # (3) classify the testX
    hat_y = []
    for i in testX.index:
        x = testX.loc[[i], :].values.T
        lnP = {k: None for k in category}
        for k in category:
            lnP[k] = np.log(prior[k]) - 0.5 * np.log(cov_det[k]) \
                     - 0.5 * u[k].T @ cov_inv[k] @ u[k] + \
                     u[k].T @ cov_inv[k] @ x - 0.5 * x.T @ cov_inv[k] @ x
            lnP[k] = lnP[k][0, 0]
        lnP = pd.DataFrame(lnP, index=['ln_prob']).T
        kind = lnP.idxmax()['ln_prob']
        hat_y.append(kind)
    hat_y = np.array([hat_y]).T

    return hat_y

if __name__ == "__main__":
    import sklearn.datasets as dataset

    breast_cancer = dataset.load_breast_cancer()
    X = breast_cancer['data'].copy()
    X = pd.DataFrame(X).apply(lambda x: (x - x.min())/(x.max() - x.min()))
    X = X.values
    train_x = X[:500, :]
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    test_x = X[500:, :]
    test_y = breast_cancer['target'][500:].reshape([test_x.shape[0], 1])
    nb = naiveBayes(train_x, train_y, test_x, lam=1)
    # Confusion matrix
    pd.crosstab(index=test_y.ravel(), columns=nb.ravel())
