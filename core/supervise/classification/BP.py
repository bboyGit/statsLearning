import numpy as np

class bp:
    """构建基于标准bp算法的神经网络分类器(待升级)"""
    def __init__(self, x, y, hiden_layer, learning_rate):
        """
        Desc: 激活函数使用sigmoid，损失函数用误差平方和。
        :param x: dataframe,训练集特征
        :param y: dataframe,训练集的值
        :param hiden_layer: A tuple of list,用于描述神经网络的隐层。例如[(1,2), (2, 3)]代表有2层隐层，
                            其中第一层有两个神经元而第二层有3个神经元。默认为0，代表没有隐层。
        :param learning_rate: 学习率
        """
        self.x = x
        self.y = y
        self.hiden_layer = hiden_layer
        self.learning_rate = learning_rate

    def fp(self, x, weight, threshold):
        """
        Desc: 在已知权重和阈值的情况下计算输出值
        :param x: 输入向量
        :param weight: 权重
        :param threshold: 阈值
        :return: 输出值向量
        """
        output_i = x
        for i in range(len(weight)):
            w = weight[i]
            thresh = threshold[i]
            input_i = output_i @ w                               # 上一层输出的线性组合作为这一次的输入
            output_i = 1/(1 + np.exp(-(input_i - thresh)))       # 将这一次输入放入sigmoid函数得到这一次的输出
        output = output_i
        return output

    def fit_zero_hiden_layer(self, sample_num, output_dim, init_weight, init_threshold):
        weight = init_weight
        threshold = init_threshold
        error = np.array([1.0] * sample_num)
        alpha = [0] * output_dim
        while True:
            for k in range(sample_num):
                # 遍历每个样本
                xk = self.x[k, :]
                yk = self.y[k, :]
                for i in range(self.x.shape[1]):
                    # 遍历每个输入
                    for j in range(output_dim):
                        # 遍历每个输出
                        yk_hat = self.fp(xk, weight=weight, threshold=threshold)
                        error[k] = 1/2 * np.sum((yk - yk_hat)**2)
                        alpha[j] = np.sum(xk * weight[0][:, j])
                        alpha_thresh = alpha[j] - threshold[0][j]
                        # 计算误差平方和对wij和theta_j的偏导
                        dw_ij = (yk_hat[j] - yk[j]) * xk[i] * np.exp(-alpha_thresh)/(1 + np.exp(-alpha_thresh))**2
                        dt_j = -(yk_hat[j] - yk[j]) * np.exp(-alpha_thresh)/(1 + np.exp(-alpha_thresh))**2
                        # 更新权重和阈值
                        weight[0][i, j] -= self.learning_rate * dw_ij
                        threshold[0][j] -= self.learning_rate * dt_j
            total_error = np.sum(error)
            print(total_error)
            if total_error <= 0.1:
                break
        return weight, threshold

    def fit_one_hiden_layer(self, sample_num, output_dim, init_weight, init_threshold):
        return

    def fit(self):
        """
        Desc: 通过梯度下降并借鉴链式法则来学习参数
        :return: 隐层和输出层的权重参数以及阈值参数
        """
        sample_num = len(self.y)                     # 训练样本量
        output_dim = self.y.shape[1]            # 输出层的维数
        input_dim = self.x.shape[1]             # 输入层的维数
        if self.hiden_layer == 0:
            # 没有隐含层
            init_weight = [np.random.uniform(0, 1, input_dim * output_dim).reshape([input_dim, output_dim])]
            init_threshold = [np.array([0.5] * output_dim)]
            params = self.fit_zero_hiden_layer(sample_num, output_dim, init_weight, init_threshold)
        elif len(self.hiden_layer) == 1:
            # 1 层隐含层
            hiden_layer_dim = self.hiden_layer[0][1]
            init_weight = [np.zeros(shape=[input_dim, hiden_layer_dim]), np.zeros(shape=[hiden_layer_dim, output_dim])]
            init_threshold = [np.array([0.5] * hiden_layer_dim), np.array([0.5] * output_dim)]
            params = self.fit_one_hiden_layer(sample_num, output_dim, init_weight, init_threshold)
        else:
            raise Exception('目前隐层的数量不能超过一层')
        return params

if __name__=='__main__':
    x = np.array([[1.3, 2], [2, 2.1], [0.8, 1.5]])
    y = np.array([[1, 0], [1, 0], [0, 1]])
    neurual_network = bp(x, y, hiden_layer=0, learning_rate=0.01)
    weight, thresh = neurual_network.fit()
