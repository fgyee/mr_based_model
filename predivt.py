# -*- coding: UTF-8 -*-
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import applications
from keras import optimizers
from keras.callbacks import TensorBoard
import numpy as np
from keras.utils import  np_utils,conv_utils
from keras.utils import plot_model
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import matplotlib
import os
from PIL import Image
import cv2
import os
matplotlib.use('Agg')

model= load_model('F:\\chess\\chesspro\\group\\group2st\\saved_models\\spleen_test_group10.hdf5')


def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (224,224))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input一张图片
    pre_x = np.array(pre_x) / 255.0
    return pre_x
#要预测的图片保存在这里
predict_dir = 'E:\\chess\\chesspro\\group\\group2st\\l-g\\group7\\train'

#这个路径下有两个文件，分别是cat和dog
test = os.listdir(predict_dir)

print(test)

images = []

for testpath in test:
    for fn in os.listdir(os.path.join(predict_dir, testpath)):
        if fn.endswith('jpg'):
            fd = os.path.join(predict_dir, testpath, fn)
            print(fd)
            images.append(fd)

pre_x = get_inputs(images)
#预测


pre_y = model.predict(pre_x)
print ('Predicted:', pre_y)
print(np.argmax(pre_y, axis=1))