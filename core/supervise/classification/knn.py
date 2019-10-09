
import numpy as np
import pandas as pd

def knn(trainX, trainY, testX, k):
    """
    Desc: Classify test set by train set in K Nearest Neighbor method
    Parameters:
      trainX: An array or a DataFrame which contains features of train set. Each row represent an observation.
      trainY: An array or a DataFrame which contains response of train set. Each row is an observation.
      testX: An array or a DataFrame which contains features of test set. Each row represent an observation.
      k: An integer indicating the number of samples in train set near a test point.
    Return: An array with 1 columns which is the class corresponding to that row.
    Note:
        KNN 是一种不需要训练的分类器，它将频率看作是后验概率，然后用最大化后验概率法则来决定一个测试集的样本该属于哪一类。
        它的哲学是“像什么就是什么”。
    """
    nobs = testX.shape[0]
    y_hat = []
    if isinstance(trainX, pd.DataFrame):
        trainX = trainX.values
    if isinstance(trainY, pd.DataFrame):
        trainY = trainY.values
    if isinstance(testX, pd.DataFrame):
        testX = testX.values

    # Iterate test data: calculate distance between the test data point to all train data and get the nearest k point.
    for i in range(nobs):
        the_x = testX[[i], :].T
        diff = trainX - the_x.T
        dist2 = diff @ diff.T
        dist2 = dist2.diagonal()
        dist2 = pd.DataFrame(dist2, columns=['d'])
        near_k = dist2.rank().sort_values('d').iloc[:k, :]
        near_idx = near_k.index
        near_x = trainX[near_idx, :]
        near_y = trainY[near_idx, :]
        mode = pd.DataFrame(near_y).mode().iloc[0, 0]
        y_hat.append(mode)

    y_hat = np.array([y_hat]).T

    return y_hat

if __name__ == "__main__":
    import sklearn.datasets as dataset
    breast_cancer = dataset.load_breast_cancer()
    train_x = breast_cancer['data'][:500, :]
    train_y = breast_cancer['target'][:500].reshape([500, 1])
    test_x = breast_cancer['data'][500:, :]
    test_y = breast_cancer['target'][500:].reshape([test_x.shape[0], 1])
    knn_hat = knn(train_x, train_y, test_x, k=20)
    # Compute Confusion matrix
    pred_real = np.concatenate([test_y, knn_hat], axis=1)
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_true=test_y, y_pred=knn_hat)

    # Compare with functions in sklearn
    import sklearn.neighbors as neighbor
    fit = neighbor.KNeighborsClassifier(20).fit(train_x, train_y.ravel())
    skhat = fit.predict(test_x)
    all(skhat == knn_hat.ravel())    # 分类结果与 scikit-learn 一致
