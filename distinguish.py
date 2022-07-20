import sys
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def main():

    #读取tmp文件夹中读取所有文件并设定大小
    filepath = 'static/tmp/'
    imgs = []
    for im in glob.glob(filepath + '*.jpg'):
        img = cv2.imread(im)
        img = cv2.resize(img, (64, 64))
        imgs.append(img)
    image = np.asarray(imgs, dtype='float32')

    # 加载模型
    model = keras.models.load_model('my_model.h5')

    # 模型预测,输出预测结果数组（待定，应当输出字符串）
    predictions = model.predict(image)

    # 输出预测结果
    outcomes = ['狗', '马', '大象', '蝴蝶', '鸡', '猫', '羊']
    result = []
    for index in range(len(predictions)):
        result.append(outcomes[predictions[index].tolist().index(1.0)])
    print(result)


if __name__ == '__main__':
    main()