import sys
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def main():
    filename = sys.argv[1]  #filename 就是用户上传的文件名
    # 加载模型
    model = keras.models.load_model('model.h5')

    filepath = 'tmp\\' + filename

    #读取文件并设定大小
    img = cv2.imread(filepath)
    img = cv2.resize(img, (100, 100))

    image = np.array(image, dtype='float32')

    # 模型预测,输出预测结果数组（待定，应当输出字符串）
    predictions = model.predict(image)
    print(predictions)


if __name__ == '__main__':
    main()