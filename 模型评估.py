from keras.models import load_model
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import random
nb_classes = 7
model = load_model('手写汉字model.hdf5')
X_test = np.load("汉字/x_test.npy",allow_pickle=True)
Y_test= np.load("汉字/y_test.npy",allow_pickle=True)
lables={'日': 0, '月': 1, '田': 2, '由': 3, '甲': 4, '申': 5, '目': 6}
print(X_test.shape)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted_classes = model.predict_classes(X_test)
# print(predicted_classes)
false=[]
ture=[]
Y_test= np.load("汉字/y_test.npy",allow_pickle=True)

for i in range(len(predicted_classes)):
    if predicted_classes[i] == Y_test[i]:
        ture.append(i)
    else:
        false.append(i)
# correct_indices = np.nonzero(predicted_classes == Y_test)[0]
# incorrect_indices = np.nonzero(predicted_classes != Y_test)[0]
# 选取9个预测正确的进行打印
print(ture)
print(false)
def turepredict():
    plt.figure(figsize=(5,5))
    plt.rc('font',family='SimHei',weight='bold',size='7')
    plt.suptitle('9张预测对的图片')
    for i in range(9):
        plt.subplot(3,3,i+1)
        j = random.randint(0,len(ture))
        plt.imshow(X_test[ture[j]].reshape(28, 28), cmap='gray', interpolation='none')
        lable2 =list(lables.keys())[list(lables.values()).index(int(Y_test[ture[j]]))]# 根据值取健
        lable1 = list(lables.keys())[list(lables.values()).index(predicted_classes[ture[j]])]  # 根据值取健
        plt.title("Predict:{} Class:{}".format(lable1, lable2))
    plt.tight_layout()
    plt.show()
# 选择9个错误的进行打印
def falsepredict():
    plt.figure(figsize=(5,5))
    plt.rc('font',family='SimHei',weight='bold',size='6')
    plt.suptitle('9张预测错的图片')
    for i in range(9):
        plt.subplot(3,3,i+1)
        j = random.randint(0,len(false))
        plt.imshow(X_test[false[j]].reshape(28, 28), cmap='gray', interpolation='none')
        lable2 =list(lables.keys())[list(lables.values()).index(int(Y_test[false[j]]))]# 根据值取健
        lable1 = list(lables.keys())[list(lables.values()).index(predicted_classes[false[j]])]  # 根据值取健
        plt.title("Predict:{} Class:{}".format(lable1, lable2))
    plt.tight_layout()
    plt.show()
while True:
    falsepredict()
    turepredict()



