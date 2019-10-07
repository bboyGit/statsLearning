
import pandas as pd
import numpy as np

def qda(trainX, trainY, testX):
    """
    Desc: Desc: Classify test set by train set in quadratic discriminant analysis
    Parameters:
      trainX: An array or a DataFrame which contains features of train set. Each row represent an observation.
      trainY: An array or a DataFrame which contains response of train set. Each row is an observation.
      testX: An array or a DataFrame which contains features of test set. Each row represent an observation.
    Return: An array with 1 columns which is the class corresponding to that row.
    Note:
        1. Assumption of Linear discriminant analysis.
          (1) Features are joint normally distributed.
        2. The algorithm in mathematical way.
          P(Y=j|X) ~ P(Y=j) * P(X|Y=j) = P(Y=j)/[pie^(1/2p) * |COVj|^0.5] * exp(-0.5(x-Uj).T @ inv(COVj) @ (x-Uj))
          ln(P(Y=j|X)) ~ ln(P(Y=j) - 0.5 * ln(|COVj|) - 0.5 * [x.T @ inv(COVj) @ x - 2Uj.T @ inv(COVj) @ x + Uj.T @ inv(COVj) @ Uj]
            ~  ln(P(Y=j))  - 0.5 * ln(|COVj|) + Uj.T @ inv(COVj) @ x -0.5 * Uj.T @ inv(cov) @ Uj - 0.5 * x.T @ inv(COVj) @ x
          So, what we need to calculate are as follows:
            (1) prior probability: P(Y=j)
            (2) feature means of each group: Uj
            (3) feature covariance matrix of each group: COVj
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

    # (2) Calculate Uj, prior probability and covariance matrix

    u = {i: None for i in category}
    prior = u.copy()
    cov, cov_inv, cov_det = u.copy(), u.copy(), u.copy()
    p = trainX.shape[1]

    for i in category:
        sub_train = train.query('y == @i').copy()
        sub_trainx = sub_train.iloc[:, :-1]
        prior[i] = sub_train.shape[0]/train.shape[0]
        ui = sub_trainx.iloc[:-1].mean().to_numpy()
        u[i] = ui.reshape(p, 1)
        cov[i] = sub_trainx.cov()
        cov_inv[i] = np.linalg.inv(cov[i])
        cov_det[i] = np.linalg.det(cov[i])

    # (3) Classify the testX
    hat = []
    for i in testX.index:
        x = testX.loc[[i], :].to_numpy().T
        lnP = {k: None for k in category}
        for k in category:
            lnP[k] = np.log(prior[k]) - 0.5 * np.log(cov_det[k]) - 0.5 * u[k].T @ cov_inv[k] @ u[k] + u[k].T @ cov_inv[k] @ x - 0.5 * x.T @ cov_inv[k] @ x
            lnP[k] = lnP[k][0, 0]
        lnP = pd.DataFrame(lnP, index=['ln_prob']).T
        kind = lnP.idxmax()['ln_prob']
        hat.append(kind)
    hat = np.array([hat]).T

    return hat


if __name__ == "__main__":
    import sklearn.datasets as dataset

    breast_cancer = dataset.load_breast_cancer()
    train_x = breast_cancer['data'][:500, :]
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    test_x = breast_cancer['data'][500:, :]
    test_y = breast_cancer['target'][500:].reshape([test_x.shape[0], 1])
    qda_hat = qda(train_x, train_y, test_x)
    # Compute Confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_true=test_y, y_pred=qda_hat)

    # Compare with sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    skqda = QuadraticDiscriminantAnalysis().fit(train_x, train_y.ravel()).predict(test_x)
    np.where(skqda != qda_hat.ravel())   # All the same