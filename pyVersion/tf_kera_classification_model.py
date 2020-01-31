import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
print(tf.__version__ )
print(sys.version_info)

for module in mpl, np,pd,sklearn,tf,keras:
    print(module.__name__, module.__version__)

# 查看版本
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()

x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(np.max(x_train),np.max(x_valid),np.max(x_test))

# x = (x-u)/std
#数据归一化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
#
##查看数据
print(np.max(x_train_scaled),np.max(x_valid_scaled),np.max(x_test_scaled))

def show_single_image(img_arr):
    plt.imshow(img_arr,cmap="binary")
    plt.show()

def show_imgs(n_rows,n_cols,x_data,y_data, classNames):
    assert len(x_data)== len(y_data)
    assert n_rows*n_cols < len(x_data)
    plt.figure(figsize=(n_cols*1.4,n_rows*1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = row*n_cols+col
            plt.subplot(n_rows,n_cols, index+1)
            plt.imshow(x_data[index],cmap="binary",interpolation="nearest")
            plt.axis('off')
            plt.title(classNames[y_data[index]])
    plt.show()

#给出相应的label名称

class_names = ['T-shirt','Trouser','Pullover','Dress','coat','Sandal','Shirt',
               'Sneaker','Bag','Ankle boot']
show_imgs(4,7,x_train,y_train,class_names)

#tf.keras.models.senquential
#建立senquential 模型方式一

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28,28]))
# model.add(keras.layers.Dense(300,activation="relu")) #全联接层
# model.add(keras.layers.Dense(200,activation="relu"))
# model.add(keras.layers.Dense(10,activation="softmax"))
#relu = max(0,x)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),#将矩阵flat
    keras.layers.Dense(300,activation="relu"),#全联接层
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
his= model.compile(loss="sparse_categorical_crossentropy", #sparse 的交叉损失熵
             optimizer="adam",
             metrics = [["accuracy","mse"]])


#Tensorboard, earlystopping, ModelCheckpoint
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)

output_model_file = os.path.join(logdir,
                                "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3), #连续5次损失函数和上一次差距小于1e-3即可停止训练

]
his = model.fit(x_train,y_train,epochs=10,
         validation_data=(x_valid,y_valid),
                 callbacks = callbacks)  #epochs 训练次数
#终端命令 tensorboard --logdir=callbacks
def plot_learning_curves(his):
    pd.DataFrame(his.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(his)
