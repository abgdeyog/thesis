from src.init_app import *
import numpy as np


# method for predicting the result for given optical flow stack
def predict(optical_flow_frames):
    arr = np.transpose(np.array(optical_flow_frames), [1, 2, 0])
    arr = arr - flow_mean
    with graph.as_default():
        features = model.predict(np.expand_dims(arr, 0))
    with graph2.as_default():
        classifier_prediction = final_classifier.predict(features)
    return classifier_prediction[0, 0]

