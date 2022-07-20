from ast import literal_eval
import os
from flask import Flask, request, redirect, url_for, render_template
import shutil

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


# 从前端接受文件并识别
@app.route('/distinguish/', methods=['POST'])
def upload_file():
    #首先清空临时文件夹
    shutil.rmtree('static/tmp')
    os.makedirs('static/tmp')

    #从前端获取文件集并保存，并将文件名保存到文件名列表中
    files = request.files.getlist('file')
    filenames = []
    for file in files:
        file.save('static/tmp/' + file.filename)
        filenames.append(file.filename)

    #调用训练模型并返回结果
    commmand = "python distinguish.py "

    #获取结果并返回
    result = os.popen(commmand).read()
    results = literal_eval(result)
    print(results)
    dicti = dict(zip(filenames, results))

    # //print(endResult)
    # //print(type(endResult))

    #filenames:图片名称列表 results:识别结果列表 一一对应
    return render_template('result.html',
                           filenames=filenames,
                           results=results,
                           dicti=dicti)


if __name__ == '__main__':
    app.run()
