import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from os import listdir
import pandas as pd

# 因为你的数据是自己的数据，所以才会有前三个方法：img2vector(), getLabel(),
# getData()用来获取数据，数据标签和处理数据。如果你用的是mnist数据集
# 或者是sklearn中自带的数据集，则直接加载即可

"""
函数说明:将28*28的二进制图像转换为1x784向量
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
       # print(lineStr)
        # 每一行的前28个元素依次添加到returnVect中
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
      #  print(int(fileNameStr.split('_')[0][1:2]))
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
        # 将每一个文件的1x784数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector(Datapath+'/%s' % (fileNameStr))
    return trainingMat

# -------------load data
# 加载数据
train_images = getData(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt')
test_images = getData(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\test全部txt')
train_labels = getLabel(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt')
test_labels = getLabel(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\test全部txt')




# -------------training
# initialize,and set cluster nums
kmeans = KMeans(n_clusters=7) #要分成的簇数也是要生成的质心数
kmeans.fit(train_images) #开始聚类
y_pred = kmeans.predict(test_images)
print(y_pred)
print(kmeans.inertia_) #聚类效果
r1 = pd.Series(kmeans.labels_).value_counts() #统计各个类别的数目

r1.rename("数量")
print(r1)
# r1.columns = list(r1.columns) + [u'类别数目'] #重命名表头
r2 = pd.DataFrame(kmeans.cluster_centers_) #找出聚类中心

r = pd.concat([r2, r1], axis = 1) #横向连接(0是纵向), 得到聚类中心对应的类别下的数目
# r = pd.concat([train_images, pd.Series(kmeans.labels_, index = test_labels)], axis = 1)  #详细
# r.columns = list(train_labels.columns) + [u'类别数目'] #重命名表头
print(r)
from sklearn.metrics import calinski_harabasz_score
print(metrics.calinski_harabasz_score(test_images, y_pred))
print(kmeans.score(train_images,test_images))
# -------------performance measure by ARI(Adjusted Rand Index)
print(metrics.adjusted_rand_score(test_labels, y_pred))