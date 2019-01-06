class Perceptron(object):
    """
    原始形态感知机
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # 步长
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        拟合函数，使用训练集来拟合模型
        :param X:training sets
        :param y:training labels
        :return:self
        """



