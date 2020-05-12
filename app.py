from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from imageai.Detection import ObjectDetection
import tensorflow as tf
# Define a flask app
app = Flask(__name__)

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath('resnet50_coco_best_v2.0.1.h5')
detector.loadModel()
custom_objects = detector.CustomObjects(dog=True,cat=True)

def detect_the_pet(img,detector):
    detections =  detector.detectCustomObjectsFromImage(custom_objects=custom_objects,input_image=img,output_image_path='result/result.png',minimum_percentage_probability=40)
    return detections

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = detect_the_pet(file_path,detector)


        
            
        return 'Result Directory'

    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    app.run()
