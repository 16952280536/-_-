import numpy as np
from cffi.backend_ctypes import xrange
from sklearn.cluster import KMeans
from sklearn import metrics
from os import listdir
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 因为你的数据是自己的数据，所以才会有前三个方法：img2vector(), getLabel(),
# getData()用来获取数据，数据标签和处理数据。如果你用的是mnist数据集
# 或者是sklearn中自带的数据集，则直接加载即可

"""
函数说明:将28*28的二进制图像转换为1x784向量
"""
testture = []
testfalse = []
ture = []
false = []
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




# for k in range(2,10):
clf = KMeans(n_clusters=7) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
clf.fit(train_images) #加载数据集合
r1 = pd.Series(clf.labels_).value_counts() #统计各个类别的数目
print(r1)
r2 = pd.DataFrame(clf.cluster_centers_) #找出聚类中心
print(r2)
# r2.rename(785)
r = pd.concat([r2, r1], axis = 1) #横向连接(0是纵向), 得到聚类中心对应的类别下的数目
r = pd.concat([r2, pd.Series(clf.labels_)], axis = 1)  #详细
r.columns =range(0,785) #重命名表头
# r.rename(columns={'0':'数量'},inplace=True)
print(r)

numSamples = len(train_images)
centroids = clf.labels_
print (centroids,type(centroids)) #显示中心点
print (clf.inertia_)  #显示聚类效果
# mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr']
mark = ['ob', 'og', 'or', 'oc', 'om', 'oy', 'ok']
#画出所有样例点 属于同一分类的绘制同样的颜色
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne.fit_transform(train_images) #进行数据降维
tsne = pd.DataFrame(tsne.embedding_, index = clf.labels_) #转换数据格式
print(tsne)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
print(r[784])
s = r[784]
df = pd.DataFrame()
print(type(tsne.iloc[1]))
n_digits = len(np.unique(train_labels))
for j in range(n_digits):
    array = []
    i=0
    for x in s:
        if(x == j):
            array.append(tsne.iloc[i])
            d=pd.DataFrame(np.array(array))  # 转换数据格式
            continue
        i=i+1
    print(d)
    # print(d[0],d[1])
    plt.plot(d[0], d[1], mark[j])



mark = ['Db', 'Dg', 'Dr', 'Dc', 'Dm', 'Dy', 'Dk']
# 画出质点，用特殊图型
centroids =  clf.cluster_centers_
for i in range(7):
    plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
    # print centroids[i, 0], centroids[i, 1]
plt.show()