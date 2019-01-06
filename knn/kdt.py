import numpy as np
from math import sqrt
# kd-tree每个结点中主要包含的数据结构如下
class Node:
    def __init__(self, data, depth=0, lchild=None, rchild=None):
        self.data = data
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild

class KdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest = None
        #self.root=self._build(dataset)

    def CreateNode(self,dataset,depth=0):
        if len(dataset)>0:
            m,n=np.shape(dataset)
            self.n=n-1
            axis=depth%self.n#每一层节点使用那个轴来划分的依据
            midIndex=int(m/2)#排序后中值的索引
            sorteddata=sorted(dataset,key=lambda x:x[axis])#对该轴/feature进行排序
            #创建节点
            node=Node(sorteddata[midIndex],depth)
            # 头部节点是深度为0的节点，也就是kdTree
            if depth==0:
                self.KdTree=node
            #左右子树
            node.lchild=self.CreateNode(dataset[:midIndex],depth+1)
            node.rchild = self.CreateNode(dataset[midIndex+1:], depth + 1)
            return node
        return None

    def pre_travelsal(self,node):
        if not node:
            return None
        if node:
            print(node.depth,node.data)
            self.pre_travelsal(node.lchild)
            self.pre_travelsal(node.rchild)
    '''
    搜索kd树
    输入：已构造的kdTree,目标点x
    输出：x的最近邻
    '''
    def search(self,x,count=1):#count:设置k值
        nearest=[]
        for i in range(count):
            nearest.append([-1, None])#dist,node
        self.nearest = np.array(nearest)

        def recurve(node):
            if node:
                axis=node.depth%self.n#划分的轴/维度
                daxis=x[axis]-node.data[axis]
                #先找到包含目标点x的叶节点
                if daxis<0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)

                #依次叶子节点作为当前最近点，计算该点与这个叶子节点的距离，并以此画圈
                dist=sqrt(sum((p1-p2)**2 for p1,p2 in zip(x,node.data)))#x与node.data都是list
                print('dist:%d and depth:%d'%(dist,node.depth))
                for i,d in enumerate(self.nearest):
                    if d[0]<0 or dist<d[0]:#对于距离还未更新过或者距离比当前任意一个小的，插入并删除最后一个
                        self.nearest=np.insert(self.nearest,i,[dist,node],axis=0)
                        self.nearest=self.nearest[:-1]
                        break

                n = list(self.nearest[:, 0]).count(-1)#还有n个最近点没算
                '''
                对最新计算的一个临近点，如果距离大于划分的父节点（node）与该点的距离,则搜索其相反子节点
                '''
                if self.nearest[-n-1, 0] > abs(daxis):
                    if daxis < 0:
                        recurve(node.rchild)
                    else:
                        recurve(node.lchild)

        recurve(self.KdTree)

        knn = self.nearest[:, 1]#k个最近的node
        belong = []
        for i in knn:
            belong.append(i.data[-1])#y class
        b = max(set(belong), key=belong.count)#得到数量最多的class

        return self.nearest, b


import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def show_train(x0,x1):
    plt.scatter(x0[:, 0], x0[:, 1], c='pink', label='[0]')
    plt.scatter(x1[:, 0], x1[:, 1], c='orange', label='[1]')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

def main():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    data = np.array(df.iloc[:100, [0, 1, -1]])
    train, test = train_test_split(data, test_size=0.1)
    x0 = np.array([x0 for i, x0 in enumerate(train) if train[i][-1] == 0])
    x1 = np.array([x1 for i, x1 in enumerate(train) if train[i][-1] == 1])

    kdt = KdTree()
    kdt.CreateNode(train)
    kdt.pre_travelsal(kdt.KdTree)

    score = 0
    for x in test:
        input('press Enter to show next:')
        show_train(x0,x1)
        plt.scatter(x[0], x[1], c='red', marker='x')  # 测试点
        near, belong = kdt.search(x[:-1], 5)  # 设置临近点的个数
        if belong == x[-1]:
            score += 1
        print("test:")
        print(x, "predict:", belong)
        print("nearest:")
        for n in near:
            print(n[1].data, "dist:", n[0])
            plt.scatter(n[1].data[0], n[1].data[1], c='green', marker='+')  # k个最近邻点
        plt.legend()
        plt.show()

    score /= len(test)
    print("score:", score)

if __name__ == '__main__':
    main()














