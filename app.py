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
    print("1ewdec")
    # #保存文件
    # file = request.files['file']
    # file.save('static/tmp/' + file.filename)
    # for file in request.files.getlist('file'):
    #     print(file)
    #调用训练模型并返回结果
    filename = file.filename

    commmand = "python distinguish.py " + filename
    result="识别失败"
    result = os.popen(commmand).read()
    # result = os.system(commmand)
    # result = "识别结果"
    print(result)
    return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run()
