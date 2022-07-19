import os
from flask import Flask, request, redirect, url_for, render_template
from regex import P
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

    #保存文件
    file = request.files['file']
    file.save('static/tmp/' + file.filename)

    #调用训练模型并返回结果
    filename = file.filename

    # commmand = "python distinguish.py " + filename
    # result = os.popen(commmand).read()
    result = "识别结果"

    return render_template('result.html', filename=filename, result=result)


if __name__ == '__main__':
    app.run()
