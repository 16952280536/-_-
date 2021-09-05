import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(5,5)
from keras.models import Sequential
from keras.utils import np_utils
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
epochs = 20
input_shape = (28, 28, 1)
nb_classes = 7 # 测试类别
X_train = np.load("汉字/x_train.npy",allow_pickle=True)
X_test = np.load("汉字/x_test.npy",allow_pickle=True)
Y_train= np.load("汉字/y_train.npy",allow_pickle=True)
Y_test= np.load("汉字/y_test.npy",allow_pickle=True)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
# 定义深度神经网络结构
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])
model.summary()
# 完成数据训练并保存训练好的模型
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

datagen.fit(X_train)

filepath = 'model.hdf5'
from keras.callbacks import ModelCheckpoint

# monitor计算每一个模型validation_data的准确率
# save_best_only 只保存最好的一个模型
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')

# steps_per_epoch指定循环次数
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=500),
                       steps_per_epoch=len(X_train)/1000, epochs=epochs,
                       validation_data=datagen.flow(X_test, Y_test, batch_size=len(X_test)),
                       validation_steps=1, callbacks=[checkpointer])
# 随着epochs的执行，模型在训练数据和测试数据集的准确率的表现情况
model.save('手写汉字model.hdf5')
history = h.history
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()