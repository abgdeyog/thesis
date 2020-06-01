from numpy.random import seed
seed(1)
import matplotlib
matplotlib.use('Agg')
import os
import scipy.io as sio
import cv2
import numpy
from flask import request
from keras.models import load_model
from tensorflow import get_default_graph
import flask
import json
from src.init_app import *
from src.db_requests import get_last_frame, get_optical_flow_stack, push_image_to_optical_flow_stack, load_new_frame
from src.optical_flow import extract_the_optical_flow
from src.classifier import predict

app = flask.Flask(__name__)
app.config["DEBUG"] = False
stack_size = 20

#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# vgg_16_weights = 'src/weights.h5'
# num_features = 4096
#
# mean_file = 'src/flow_mean.mat'
# d = sio.loadmat(mean_file)
# flow_mean = d['image_mean']
#
# # initializing the cnn
# model, graph = init_cnn()
#
# # loading saved classifier
# final_classifier = load_model("src/classifier_100")
# graph2 = get_default_graph()



# api endpoint for uploading the frames from image stream
@app.route('/load_frame', methods=['POST'])
def load_frame():
    global model
    global prvs_image
    source_id = request.form.get("id",type=int, default=None)
    frame_new = request.files.get("frame_1", "").read()
    frame_new = numpy.fromstring(frame_new, numpy.uint8)
    frame_prvs = get_last_frame(source_id)
    load_new_frame(frame_new, source_id)
    # convert numpy array to image
    frame_new = cv2.imdecode(frame_new, cv2.IMREAD_ANYCOLOR)
    if frame_prvs is None:
        frame_previous = frame_new
    else:
        frame_previous = cv2.imdecode(frame_prvs, cv2.IMREAD_ANYCOLOR)
    optical_flow_x, optical_flow_y = extract_the_optical_flow(frame_new, frame_previous)
    stack_length = push_image_to_optical_flow_stack(optical_flow_x, optical_flow_y, source_id)
    return app.response_class(
        status=200,
        mimetype='application/json',
        response=json.dumps({"optical_flow_frames": stack_length})
    )

# api endpoint for predicting the fall for currently processed images sequence
@app.route('/make_prediction', methods=['GET'])
def get_prediction():
    id = request.args.get('id', type=int, default=None)
    # request the optical flow stack from db and call the predict() function with this stack
    prediction = predict(get_optical_flow_stack(id))
    if prediction > 0.5:
        result = "Not Fall"
    else:
        result = "Fall"
    return app.response_class(
        status=200,
        mimetype='application/json',
        response=json.dumps({"prediction": result, "raw_prediction": str(prediction)})
    )


app.run()
