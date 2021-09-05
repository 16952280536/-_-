from random import random

import numpy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
from numpy import *
from sklearn.cluster import KMeans
from sklearn import metrics
from os import listdir
import pandas as pd
# 因为你的数据是自己的数据，所以才会有前三个方法：img2vector(), getLabel(),
# getData()用来获取数据，数据标签和处理数据。如果你用的是mnist数据集
# 或者是sklearn中自带的数据集，则直接加载即可





def distEclud(vecA, vecB):
    '''
    计算两个向量的欧氏距离
    param vecA,vecB: 两个待计算距离的向量  numpy.ndarray
    return : 两个向量的欧氏距离
    '''
    # 马氏距离
    # from scipy.spatial.distance import mahalanobis
    # cov = np.cov(vecB, rowvar=False)
    # avg_distri = np.average(vecB, axis=0)
    # dis = mahalanobis(vecA, avg_distri, cov)
    # return dis


    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    '''
    随机挑选k个初始簇中心
    param dataSet: 数据矩阵
    param k: 簇数
    return : k个初始簇中心
    '''
    n = shape(dataSet)[1]  # 7
    centroids = mat(zeros((k, n)))
    for j in range(n):
        try:
            minJ = min(dataSet[:, j])
        except:
            print(dataSet)
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def Kmeans(dataSet, k, distMeas=distEclud, createCent=randCent):

    '''
    K-均值聚类算法
    param dataSet: 数据集  numpy.matrix
    param k: 类别数        int
    param distMeas: 计算距离的方法,默认为欧氏距离  function
    param createCent: 产生初始簇中心的方式,默认是随机生成 function
    return centroids: 聚类结果的类簇中心 numpy.matrix
    return clusterAssment: 每条记录的聚类结果 numpy.matrix matrix([[clusterTag,欧氏距离平方],
                                                                [clusterTag,欧氏距离平方],...])
    '''
    m = np.shape(dataSet)[0]  # 样本数m
    clusterAssment = np.mat(np.zeros([m, 2]))  # m*2, 第一列记录样本属于哪一类,第二列记录样本到类中心点的距离
    centroids = createCent(dataSet, k)  # k*特征数,中心点的坐标
    clusterChanged = True
    r=0
    while clusterChanged:
        r=r+1
        print("第",r,"次")
        clusterChanged = False
        #         对每一个样本进行循环
        for i in range(m):
            # 计算样本距离每一个中心的距离,保留最小距离和归类
            minDist = float("inf")
            minIndex = -1
            for j in range(k):
                distIJ = distMeas(dataSet[i, :], centroids[j, :])
                if distIJ < minDist:
                    minDist = distIJ
                    minIndex = j

            # 只要有一个样本的归类发生变化,标记就改为True(继续循环)
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            # 把 归类和最小距离存到clusterAssment中
            clusterAssment[i, :] = minIndex, minDist ** 2
            # print("第",i,"点最小距离:",clusterAssment[i,1])
        # 每一轮打印中心点的坐标
        print(centroids)

        # 更新中心点坐标(根据样本的归类,求每一个分类的质心)
        for cent in range(k):
            pstInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pstInClust, axis=0) # 压缩行，对各列求均值，返回 1* n 矩阵

    return centroids, clusterAssment
"""
函数说明:将28x28的二进制图像转换为1x784向量
"""
def img2vector(filename):
    # 创建1x784零向量
    returnVect = np.zeros((1, 784))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(28):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(28):
            returnVect[0, 28 * i + j] = float(lineStr[j])
    # 返回转换后的1x784向量
    return returnVect

'''
函数说明：获取标签
'''
def getLabel(Datapath):
    # 训练集的Labels
    hwLabels = []
    # 返回Datapath目录下的文件名
    trainingFileList = listdir(Datapath)
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0][1:2])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
    return hwLabels

'''
函数说明：获取数据
'''
def getData(Datapath):
    # 返回train目录下的文件名
    trainingFileList = listdir(Datapath)
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,训练集
    trainingMat = np.zeros((m, 784))
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 将每一个文件的784*784数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector(Datapath+'/%s' % (fileNameStr))
    return trainingMat

# -------------load data
# 加载数据
train_images = getData(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt')
test_images = getData(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\test全部txt')
train_labels = getLabel(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt')
test_labels = getLabel(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\test全部txt')
yi=0
er=0
san=0
zer=0
si=0
wu=0
liu=0
m=len(train_labels)
print(train_labels[0])
print(train_labels[1])
for s in range (m):
    if train_labels[s]==0:
        zer=zer+1
        continue
    elif train_labels[s]==1:
        yi=yi+1
        continue
    elif train_labels[s]==2:
        er=er+1
        continue
    elif train_labels[s]==3:
        san=san+1
        continue
    elif train_labels[s]==4:
        si=si+1
        continue
    elif train_labels[s]==5:
        wu=wu+1
        continue
    else:
        liu=liu+1
data = scale(train_images)
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)
n_digits = len(np.unique(train_labels))
# 降维和k均值聚类
reduced_data = PCA(n_components=50).fit_transform(data) #降2维
reduced_data = TSNE(n_components=2).fit_transform(reduced_data)
center = Kmeans(reduced_data,7)
print(center[0])
test_images = PCA(n_components=50).fit_transform(scale(test_images))  #降2维
test_images = TSNE(n_components=2).fit_transform(test_images)
kmeans = KMeans(init=center[0], n_clusters=7, n_init=10,) # 7为分的族数
kmeans.fit(reduced_data)
r1 = pd.Series(kmeans.labels_).value_counts() #统计各个类别的数目

print(r1)

worry = abs(zer-r1.loc[0])+abs(yi-r1.loc[1])
worry+=abs(er-r1.loc[2])+abs(san-r1.loc[3])
worry+=abs(si-r1.loc[4])+abs(wu-r1.loc[5])
worry+=abs(liu-r1.loc[6])
accurcent = ((m-worry)/float(m))
print(accurcent)


label_pred = kmeans.labels_
y_pred = kmeans.predict(test_images)
print("兰德系数：")
print(metrics.adjusted_rand_score(test_labels, y_pred))
plt.clf()



# 画簇心和点
centroids = kmeans.cluster_centers_ # 中心点
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='.', s=169, linewidths=3,
            color='b', zorder=10)
#[[-15.492258   42.903057 ], [ 56.887783   -5.064375 ], [ 17.557386  -41.899414 ], [-69.151024   10.224685 ], [-41.16663   -46.208748 ], [ -3.8927143  -1.8861859], [ 33.02237    31.524681 ]]
color_list = ['#000080', '#006400', '#00CED1', '#800000', '#800080',
              '#CD5C5C', '#DAA520', '#E6E6FA', '#F08080', '#FFE4C4']




for i in range(n_digits):
    x = reduced_data[label_pred == i]
    print(x)
    # print(reduced_data)
    plt.scatter(x[:, 0], x[:, 1], c=color_list[i], marker='*', label='label%s' % i)
plt.title('K-means')
plt.axis('on')
plt.show()