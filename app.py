import os
from uuid import uuid4
from flask import Flask, render_template, request, send_from_directory
import cv2 as cv
import numpy as np
from sklearn.externals import joblib
import base64
import re
import model.load as ml

__author__ = 'sarahawwad'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload/", methods=['GET','POST'])
def upload():

    imgData = request.get_data()
    convertImage(imgData)
    img = cv.imread("output.png")
    img = ml.preprocess(img)
    img = ml.gabor(img)
    img = img.flatten()
    img = np.array([img])
    print(img.shape)

    selectionname = "crossvalidation_models/svm_gabor_selection.joblib"
    modelname = "crossvalidation_models/svm_gabor.joblib"

    loaded_model = joblib.load(selectionname)
    imgnew = loaded_model.transform(img)

    loaded_model = joblib.load(modelname)
    res = loaded_model.predict(imgnew)
    res = res[0]
    names = ["fracture","osteoarthritis", "elbow dislocation"]
    print("Result = ", names[res])
    print("classification")
    return names[res]

# @app.route('/upload/<filename>')
# def send_image(filename):
#     return send_from_directory("images", filename)

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
      output.write(base64.b64decode(imgstr))

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)


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