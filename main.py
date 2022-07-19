import os
from bs4 import ResultSet
from flask import Flask, request, redirect, url_for, render_template
from regex import P

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


# 从前端接受文件并保存到本地
@app.route('/distinguish/', methods=['POST', 'GET'])
def upload_file():
    file = request.files['file']
    # file.save('tmp/' + file.filename)

    #调用训练模型并返回结果
    filename = file.filename

    commmand = "python distinguish.py " + filename
    # result = os.popen(commmand).read()
    result = "识别结果"
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run()
