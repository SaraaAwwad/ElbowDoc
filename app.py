import os
from uuid import uuid4
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, session, abort,url_for
import cv2 as cv
import numpy as np
from sklearn.externals import joblib
import base64
import re
import json
import model.load as ml
import MySQLdb
from shutil import copyfile
from random import *
import shutil

__author__ = 'sarahawwad'

app = Flask(__name__)

conn = MySQLdb.connect(host="localhost",user="root",password="",db="elbow")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print("test1")

folders = ["fractures", "osteo", "r_elbow_dislocation"]
names = ["fracture", "osteoarthritis", "dislocation"]


# @app.route("/")
# def index():
#     return render_template("login.html")

@app.route('/')
def login():
    return render_template("login.html")

@app.route('/check',methods=["GET","POST"])
def check():
    username = str(request.form["user"])
    password = str(request.form["password"])
    cursor = conn.cursor()
    cursor1 = conn.cursor()
    cursor.execute("SELECT username FROM user WHERE username ='"+username+"' AND password ='"+password+"'")
    cursor1.execute("SELECT type_id_fk FROM user WHERE username ='"+username+"' AND password ='"+password+"'")
    user = cursor.fetchone()
    type = cursor1.fetchone()

    # if len(user) is 1:
    #     return render_template("home.html",data=type[0])
    # else:
    #     return "failed"

    if len(user) is 1:
        if type[0] == 2:
            return render_template("upload.html")
        else:
            return render_template("doctor.html")
    else:
        return "failed"

@app.route("/classify", methods=['GET','POST'])
def classify():
    return render_template("upload.html")

@app.route("/upload/", methods=['GET','POST'])
def upload():
    print("test")
    imgData = request.get_data()
    convertImage(imgData)
    img = cv.imread("output.png")
    img = ml.preprocess(img)
    img = ml.gabor(img)
    img = img.flatten()
    img = np.array([img])
    print(img.shape)

    selectionname = "svm_gabor_selection.joblib"
    modelname = "svm_gabor.joblib"

    loaded_model = joblib.load(selectionname)
    imgnew = loaded_model.transform(img)

    loaded_model = joblib.load(modelname)
    res = loaded_model.predict(imgnew)
    res = res[0]

    p = np.array(loaded_model.decision_function(imgnew))  # decision is a voting function
    prob = np.exp(p) / np.sum(np.exp(p), axis=1)  # softmax after the voting
    classes = loaded_model.predict(imgnew)

    print("second")
    _ = [print('Sample={}, Prediction={},\n Votes={} \nP={}, '.format(idx, c, v, s)) for idx, (v, s, c) in
         enumerate(zip(p, prob, classes))]
    probability = prob[:, classes]

    percentage = "{:.1f}%".format(probability[0][0] * 100.0)
    print(percentage)
    names = ["fracture","osteoarthritis", "elbow dislocation"]
    print("Result = ", names[res])
    print("classification")
    variable = [names[res],percentage]

    return jsonify(variable)

@app.route("/retrain/", methods=['GET','POST'])
def retrain():
    #img = request.form['img']
    #convertImage(img)
    choice = request.form['choice']
    print("choice = ", choice)

    # y_new = [int(choice)]
    # modelname = "sgd.joblib"
    # loaded_model = joblib.load(modelname)
    # loaded_model.partial_fit(img, y_new)
    # joblib.dump(loaded_model, 'sgd2.joblib')
    # print("size fit =", os.path.getsize('sgd.joblib'))
    # print("size fit =", os.path.getsize('sgd2.joblib'))
    #
    print("test retrain")
    x = randint(1, 100)
    dest = "hybrid2/"+ folders[int(choice)] +"/" +str(x) +".png"
    print("dest = ", dest)
    shutil.copy2('output.png', dest)  # target filename is /dst/dir/file.ext
    ml.retrain()

    return "success"

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
      output.write(base64.b64decode(imgstr))


if __name__ == "__main__":
    # app.run(port=4555, debug=True)
    port = int(os.environ.get('PORT', 4555))
    app.run(host='127.0.0.1', port=port)


# #app = Flask(__name__)
# app = Flask(__name__, static_folder="images")
#
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#
# @app.route("/")
# def index():
#     return render_template("upload.html")
#
# @app.route("/upload", methods=["POST"])
# def upload():
#     target = os.path.join(APP_ROOT, 'images/')
#     # target = os.path.join(APP_ROOT, 'static/')
#     print(target)
#     if not os.path.isdir(target):
#             os.mkdir(target)
#     else:
#         print("Couldn't create upload directory: {}".format(target))
#     print(request.files.getlist("file"))
#     for upload in request.files.getlist("file"):
#         print(upload)
#         print("{} is the file name".format(upload.filename))
#         filename = upload.filename
#         destination = "/".join([target, filename])
#         print ("Accept incoming file:", filename)
#         print ("Save it to:", destination)
#         upload.save(destination)
#
#     # return send_from_directory("images", filename, as_attachment=True)
#     return render_template("complete.html", image_name=filename)
#
# @app.route('/upload/<filename>')
# def send_image(filename):
#     return send_from_directory(app.static_folder, filename)
#
# if __name__ == "__main__":
#     app.run(port=4555, debug=True)


#
# __author__ = 'ibininja'
#
# app = Flask(__name__)
#
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#
# @app.route("/")
# def index():
#     return render_template("upload.html")
#
# @app.route("/upload", methods=['POST'])
# def upload():
#     target = os.path.join(APP_ROOT, 'images/')
#     print(target)
#
#     if not os.path.isdir(target):
#         os.mkdir(target)
#
#     for file in request.files.getlist("file"):
#         print(file)
#         filename = file.filename
#         destination = "/".join([target, filename])
#         print(destination)
#         file.save(destination)
#
#     return render_template("complete.html")
#
# if __name__ == "__main__":
#     app.run(port=4555, debug=True)