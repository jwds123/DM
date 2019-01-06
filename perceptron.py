import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    原始形态感知机
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        拟合函数，使用训练集来拟合模型
        :param X:training sets
        :param y:training labels
        :return:self
        """
        # X's each col represent a feature
        # initialization wb(weight plus bias)
        self.wb = np.zeros(1 + X.shape[1])
        # the main process of fitting
        self.errors_ = []  # store the errors for each iteration
        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))#yi-predict(xi)!=0:误分类
                #update = self.eta * (yi)
                self.wb[1:] += update * xi
                self.wb[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, xi):
        """
        计算净输入
        :param xi:
        :return:净输入
        """
        #w*x1+b
        return np.dot(xi, self.wb[1:]) + self.wb[0]

    def predict(self, xi):
        """
        计算预测值
        :param xi:
        :return:-1 or 1
        """
        #sign(wx+b)-->yi
        return np.where(self.net_input(xi) <= 0.0, -1, 1)

    def plot_decision_regions(self, X, y, resolution=0.02):
        """
        拟合效果可视化
        :param X:training sets
        :param y:training labels
        :param resolution:分辨率
        :return:None
        """
        # initialization colors map
        colors = ['red', 'blue']
        markers = ['o', 'x']
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision regions
        x1_max, x1_min = max(X[:, 0]) + 1, min(X[:, 0]) - 1
        x2_max, x2_min = max(X[:, 1]) + 1, min(X[:, 1]) - 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
        plt.show()

'''
对偶形式：
f(x)=sign(sum(aj*yj*xj.xi+b))
判断条件，误分类：
yi*(sum(aj*yj*xj.xi+b))<=0
'''
class DualPerceptron(object):
    def __init__(self, eta=0.01):
        self.eta = eta
        #self.n_iter = n_iter
        self.a=None
        self.b=None
        self.G_matrix=None


    # 计算Gram Matrix
    def calculate_g_matrix(self,data):
        #x的每个实例xi=[xi1,xi2]--2 features
        self.G_matrix = np.zeros((data.shape[0], data.shape[0]))
        # 填充Gram Matrix
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                #G_matrix[i][j] = np.sum(data[i, 0:-1] * data[j, 0:-1])
                self.G_matrix[i][j] = np.sum(data[i]*data[j])
        return self

    # 迭代的判定条件
    def judge(self,X, y, index):
        #global a,b,G_matrix
        tmp = 0

        for m in range(X.shape[0]):

            tmp += self.a[m] * y[m] * self.G_matrix[index][m]
        res=(tmp + self.b) * y[index]

        return res


    def fit(self,X,y):
        """
        对偶形态的感知机
        由于对偶形式中训练实例仅以内积的形式出现
        因此，若事先求出Gram Matrix，能大大减少计算量
        :param data:训练数据集;ndarray object
        :return:w,b
        """
        # 计算Gram_Matrix
        self.calculate_g_matrix(X)

        # 读取数据集中含有的样本数
        num_samples = X.shape[0]
        # 读取数据集中特征向量的个数
        num_features = X.shape[1]
        # 初始化a,b。a=ni*eta
        self.a, self.b = [0] * num_samples, 0
        # 初始化weight
        self.w = np.zeros((1, num_features))
        #M是误分类项的迭代次数矩阵，迭代次数越多说明该项距离超平面越近，对分类结果影响就越大
        '''
        当使用feature 0,1时，效果不好
        第1个值第277次迭代
        第69个值第199次迭代
        第1个值第278次迭代
        第69个值第200次迭代
        第11个值第100次迭代

        '''
        M=[0]*num_samples
        i = 0
        while i < num_samples:
            if self.judge(X, y, i) <= 0:
                M[i] += 1
                print('第%d个值第%d次迭代'%(i,M[i]))
                self.a[i] += self.eta
                self.b += self.eta*y[i]
                i = 0

            else:
                i += 1

        for j in range(num_samples):
            self.w += self.a[j] * X[j] * y[j]#有两列，w.shape=(1,2)


        return self
    def net_input(self, xi):
        """
        计算净输入
        :param xi:
        :return:净输入
        """
        #w*x1+b
        return np.dot(xi, self.w.reshape(2,1)) + self.b

    def predict(self, xi):
        """
        计算预测值
        :param xi:
        :return:-1 or 1
        """
        #sign(wx+b)-->yi
        return np.where(self.net_input(xi) <= 0.0, -1, 1)

    def plot_decision_regions(self, X, y, resolution=0.02):
        """
        拟合效果可视化
        :param X:training sets
        :param y:training labels
        :param resolution:分辨率
        :return:None
        """
        # initialization colors map
        colors = ['red', 'blue']
        markers = ['o', 'x']
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision regions
        x1_max, x1_min = max(X[:, 0]) + 1, min(X[:, 0]) - 1
        x2_max, x2_min = max(X[:, 1]) + 1, min(X[:, 1]) - 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        #m=np.array([xx1.ravel(), xx2.ravel()]).T
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
        plt.show()


def main():
    iris = load_iris()
    #X = iris.data[:100, [1,2]]
    # X = iris.data[:100, [0,1]]
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)
    # ppn = Perceptron(eta=0.1, n_iter=10)
    # ppn.fit(X_train, y_train)
    # ppn.plot_decision_regions(X_test, y_test)
    dppn = DualPerceptron(eta=0.1)
    dppn.fit(X_train, y_train)
    dppn.plot_decision_regions(X_test, y_test)



if __name__ == '__main__':
    main()



