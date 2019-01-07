import math

class NaiveBayes:
    def __init__(self):
        self.model = None

    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))#特征列向量

    def stdev(self,X):
        m=self.mean(X)
        return math.sqrt(sum([math.pow(x-m,2) for x in X])/float(len(X)))

    def get_mean_std(self,data):
        #data=[x1,x2...,xn],其中xi是一个实例。i in zip(*data) 就是每个特征向量了
        return [[self.mean(i),self.stdev(i)] for i in zip(*data)]

    #计算每个服从高斯分布的实例的条件概率分布
    def gaussianprob(self,x,mean,std):
        #传入参数必须是scalar
        #exp=math.exp(-(x-mean)**2/(2*std**2))
        expo=math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
        return 1/math.sqrt(2*math.pi*math.pow(std,2))*expo

    #计算模型参数
    def fit(self,X,y):
        #train_data={label:[para]} y:[mean,std]
        '''
        对于不同类别分别计算mean与std
        :param X:
        :param y:
        :return:
        '''
        labels=list(set(y))
        data={label:[] for label in labels}
        for x,label in zip(X,y):
            data[label].append(x)
        self.model={label:self.get_mean_std(x) for label,x in data.items()}
        return "get model parameter"

    #计算后验概率
    def post_prob(self,input_data):
        probs={}
        for label,value in self.model.items():
            probs[label]=1
            for i in range(len(value)):
                mean,std=value[i]
                probs[label]*=self.gaussianprob(input_data[i],mean,std)
        return probs

    def predict(self,X_test):
        #{0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26} {class:prob}
        #根据列表后一个值排序，然后最后一个值作为class
        label=sorted(self.post_prob(X_test).items(),key=lambda x:x[-1])[-1][0]
        return label

    def score(self,X_test,y_test):
        right=0
        for X,y in zip(X_test,y_test):
            label=self.predict(X)
            if label==y:
                right+=1
        score=right/float(len(X_test))
        print(score)
        return score


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = NaiveBayes()
    model.fit(X_train,y_train)
    print(model.predict([4.4, 3.2, 1.3, 0.2]))
    model.score(X_test, y_test)


if __name__ == '__main__':
    main()

