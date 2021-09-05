# coding:utf-8
import numpy as np
import random
#  os 模块中导入函数listdir，该函数可以列出给定目录的文件名
from os import listdir
import operator
import matplotlib.pyplot as plt
testture = []
testfalse = []
ture = []
false = []
def  img2vector(filename):
    """实现将图片转换为向量形式"""
    return_vector = np.zeros((1, 784))
    fr = open(filename)
    for  i  in  range(28):
        line = fr.readline()
        for  j  in  range(28):
            return_vector[0, 28 * i + j] = int(line[j])
    # print(return_vector.shape)
    return return_vector
#  inX 用于分类的输入向量
#  dataSet表示训练样本集
#  标签向量为labels，标签向量的元素数目和矩阵dataSet的行数相同
#  参数k表示选择最近邻居的数目
def  classify0(inx,data_set, labels, k):
    """实现k近邻"""
    diff_mat = inx - data_set     #  各个属性特征做差
    sq_diff_mat = diff_mat ** 2    #  各个差值求平方
    sq_distances = sq_diff_mat.sum(axis=1)    #  按行求和
    distances = sq_distances ** 0.5     #  开方
    sorted_dist_indicies = distances.argsort()    #  按照从小到大排序，并输出相应的索引值
    class_count = {}    #  创建一个字典，存储k个距离中的不同标签的数量
    for  i  in  range(k):
        vote_label = labels[sorted_dist_indicies[i]]    #  求出第i个标签
        #  访问字典中值为vote_label标签的数值再加1，
        #  class_count.get(vote_label, 0)中的0表示当为查询到vote_label时的默认值
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    #  将获取的k个近邻的标签类进行排序
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
    #  标签类最多的就是未知数据的类
    return sorted_class_count[0][0]
def  hand_writing_class_test():
    """手写数字KNN分类"""
    hand_writing_labels = []    #  手写数字类别标签
    training_file_list = listdir(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt')    #  获得文件中目录列表，训练数据集
    m = len(training_file_list)     #  求得文件中目录文件个数（训练数据集）
    training_mat = np.zeros((m, 784))    #  创建训练数据矩阵，特征属性矩阵
    for  i  in  range(m):
        file_name_str = training_file_list[i]    #  获取单个文件名
        file_str = file_name_str.split(' ')[0]    #  将文件名中的空字符去掉，这里的[0]是将文件名取出来
        # print(file_str)
        class_num_str = int(file_str[1:2])    #  取出数字类别
        # print(class_num_str)
        hand_writing_labels.append(class_num_str)    #  将数字类别添加到类别标签矩阵中
        #  将图像格式转换为向量形式
        filename=r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt\%s'  %  file_name_str
        training_mat[i, :] = img2vector(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\train全部txt\%s'  %  file_name_str)
        test_file_list = listdir(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\test全部txt')    #  获得文件中目录列表，测试数据集
    x=[1, 2, 3, 4 , 5, 6, 7, 8, 9, 10]
    y=[]
    # print(training_mat[0].shape)
    for k in x: # k值不同
        error_count = 0  # 错误分类个数
        m_test = len(test_file_list)  # 测试数据集个数
        for  i  in  range(m_test):
            file_name_str = test_file_list[i]    #  获取单个文件名（测试数据集）
            file_str = file_name_str.split('.')[0]    #  将文件名中的空字符去掉，这里的[0]是将文件名取出来（测试数据集）
            class_num_str = int(file_str[1:2])     #  取出数字类别（测试数据集）
            #  将图像格式转换为向量形式（测试数据集）
            vector_under_test = img2vector(r'C:\Users\Lenovo\Desktop\code\人工智能\汉字识别\test全部txt\%s'  %  file_name_str)
            #  KNN分类，以测试数据集为未知数据，训练数据为训练数据
            classifier_result = classify0(vector_under_test, training_mat, hand_writing_labels, k)
            #  输出分类结果和真实类别
            print('the classifier came back with: %d, the real answer is: %d'  %  (classifier_result, class_num_str))
            #  计算错误分类个数
            if  classifier_result  !=  class_num_str:
                error_count  +=  1
                if k==6:
                    false.append([class_num_str,classifier_result])
                    testfalse.append(vector_under_test)
            else:
                if k==6:
                    ture.append([class_num_str,classifier_result])
                    testture.append(vector_under_test)

    #  输出错误分类个数和错误率
        print("\n 错误总数: %d"  %  error_count)
        print("\n 总错误率: %f"  %  (error_count / float(m_test)))
        y.append(1-error_count / float(m_test))
    return x,y
#  调用手写识别
x,y=hand_writing_class_test()
for i in range(len(x)):
    print("\nk值为%d时" %(x[i]))
    print("总正确率:%f" % (y[i]))
# 进行结果可视化
plt.figure()
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
x = np.array(x)
y = np.array(y)
plt.plot(x, y)
plt.title("k与准确率的关系走势图")
plt.xlabel("k值")
plt.ylabel("准确率")
plt.savefig("./k值对准确率的影响.png")
plt.show()

lables={'日': 3, '月': 7, '田': 5, '由': 6, '甲': 1, '申': 4, '目': 2}
# 输出k=6时，9个错误预估的图像
def showimg(test,lable):
    plt.figure(figsize=(5,5))
    plt.rc('font',family='SimHei',weight='bold',size='6')
    for i in range(9):
        plt.subplot(3, 3, i+1)
        j=random.randint(0,len(lable))
        imgdata=test[j]
        plt.imshow(imgdata.reshape(28,28), cmap='gray', interpolation='none')
        lable2 =list(lables.keys())[list(lables.values()).index((lable[j][0]))]# 根据值取健
        lable1 = list(lables.keys())[list(lables.values()).index(lable[j][1])]  # 根据值取健
        plt.title("Predict:{} Class:{}".format(lable1, lable2))
        # plt.title(y_train[j])
    plt.suptitle('K=6')
    plt.tight_layout()
    plt.show()
# 不断显示k值为6时的9个正确预估与错误预估
showimg(testfalse,false)
showimg(testture,ture)

