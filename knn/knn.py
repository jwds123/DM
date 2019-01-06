import math
class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
                parameter: n_neighbors 临近点个数
                parameter: p 距离度量
                """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X):
        # 取出距离最近的n个点的索引和y
        knn_list=[]
        # 先取出n个点
        for i in range(self.n):
            dist=np.linalg.norm(X-self.X_train[i],self.p)
            knn_list.append((dist,self.y_train[i]))
        # 对于后面的点，删除n+1个点中的最大值
        for i in range(self.n,len(self.X_train)):
            max_index=knn_list.index(max(knn_list,key=lambda x:x[0]))#按照dist大小找出目前dist最大值下标
            dist = np.linalg.norm(X - self.X_train[i], self.p)
            if knn_list[max_index][0]>dist:
                knn_list[max_index]=(dist,self.y_train[i])

        #统计不同类别出现的次数，次数最多的那个就是该实例的类别
        classcount = [k[-1] for k in knn_list]#knn_list是个list不能knn_list[1
        votedict=Counter(classcount)#yi:count
        classpredict=sorted(votedict,key=lambda x:x)[-1]#取最大的
        return classpredict

    def score(self,X_test,y_test):
        count=0
        for X,y in zip(X_test,y_test):
            label=self.predict(X)
            if label==y:
                count+=1
        return count/len(X_test)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter

def main():
    # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = KNN(X_train, y_train)
    print(clf.score(X_test, y_test))
    test_point = X_test[10]
    print('Test Point: {}'.format(clf.predict(test_point)))

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0], test_point[1], 'ro', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
