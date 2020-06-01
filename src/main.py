from numpy.random import seed
from sklearn.model_selection import train_test_split
seed(1)
import numpy as np
import cv2 as cv
import os
import matplotlib
from keras.models import load_model
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import cv2
import glob
import gc

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                          Activation, Dense, Dropout, ZeroPadding2D)

from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
vgg_16_weights = 'weights.h5'
num_features = 4096
# ========================================================================
# Optical Flow Extractor
# ========================================================================
def extract_the_optical_flow(video_path, output_path):
    frames_names = sorted(os.listdir(video_path))
    frames = []
    frame_count = 0
    for frame_name in frames_names:
        frames.append(cv.imread(video_path+'/'+frame_name))
        #cv2.imwrite('optical_flow_images/'+frame_name, frames[-1])
        # cv2.imshow(frame_name, frames[-1])
    # params for ShiTomasi corner detection
    frame1 = frames[frame_count]
    frame_count += 1
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv_x = np.zeros_like(frame1)
    hsv_x[..., 1] = 0
    hsv_x[..., 2] = 0

    hsv_y = np.zeros_like(frame1)
    hsv_y[..., 1] = 0
    hsv_y[..., 2] = 0
    out_frame_n = 0
    while frame_count < len(frames):
        frame2 = frames[frame_count]
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        # hsv_x[...,0] = ang*180/np.pi/2
        # hsv_y[...,0] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        # cv.imshow('frame0_x', flow[..., 0])
        # cv.imshow('frame1_x', flow[..., 1])
        #print(len(flow[..., 0]))
        # im = Image.fromarray(arr)
        gray_x = np.array(flow[..., 0])
        gray_x_to_write = cv.convertScaleAbs(gray_x, alpha=255.0)
        gray_y = np.array(flow[..., 1])
        gray_y_to_write = cv.convertScaleAbs(gray_y, alpha=255.0)
        frame_number_str = ''.join(['0' for _ in range(4 - len(str(out_frame_n)))]) + str(out_frame_n)
        cv.imwrite(output_path + '/flow_x_' + frame_number_str + '.jpg', gray_x_to_write)
        cv.imwrite(output_path + '/flow_y_' + frame_number_str + '.jpg', gray_y_to_write)
        out_frame_n += 1
        # cv.imshow('frame2_y', flow[..., 1])
        # cv.imwrite('optical_flow_images/'+frames_names[frame_count]+'1', flow[..., 1])
        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv.imwrite('opticalfb.png',frame2)
        #     cv.imwrite('opticalhsv.png',bgr)
        frame_count += 1
        prvs = next


video_path = 'raw_notfall-01'
optical_flow_path = 'optical_flow_images'
extract_the_optical_flow(video_path, optical_flow_path)
# ========================================================================
# VGG-16 ARCHITECTURE
# ========================================================================
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(num_features, name='fc6', kernel_initializer='glorot_uniform'))

# ========================================================================
# WEIGHT INITIALIZATION
# ========================================================================
layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
               'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
               'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
h5 = h5py.File(vgg_16_weights, 'r')

layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Copy the weights stored in the 'vgg_16_weights' file to the
# feature extractor part of the VGG16
for layer in layerscaffe[:-3]:
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    # w2 = np.transpose(np.asarray(w2), (0,1,2,3))
    # w2 = w2[:, :, ::-1, ::-1]
    w2 = np.transpose(np.asarray(w2), [2, 3, 1, 0])
    w2 = w2[::-1, ::-1, :, :]
    b2 = np.asarray(b2)
    # layer_dict[layer].W.set_value(w2)
    # layer_dict[layer].b.set_value(b2)
    layer_dict[layer].set_weights((w2, b2))
# sys.exit()
# Copy the weights of the first fully-connected layer (fc6)
layer = layerscaffe[-3]
w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
w2 = np.transpose(np.asarray(w2), [1, 0])
b2 = np.asarray(b2)
# layer_dict[layer].W.set_value(w2)
# layer_dict[layer].b.set_value(b2)
layer_dict[layer].set_weights((w2, b2))
layer = layerscaffe[-3]
w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
w2 = np.transpose(np.asarray(w2), [1, 0])
b2 = np.asarray(b2)
layer_dict[layer].set_weights((w2, b2))

mean_file = 'flow_mean.mat'
d = sio.loadmat(mean_file)
flow_mean = d['image_mean']
video_path = "optical_flow_images"
x_images = glob.glob(video_path + '/flow_x*.jpg')
x_images.sort()
y_images = glob.glob(video_path + '/flow_y*.jpg')
y_images.sort()


############################## SOME SHIT HAPPENS HERE
L = 10
nb_stacks = 10
flow = np.zeros(shape=(224, 224, 2 * L, nb_stacks), dtype=np.float64)


def generator(list1, lits2):
    for x, y in zip(list1, lits2):
        yield x, y


gen = generator(x_images, y_images)
for i in range(len(x_images)):
    flow_x_file, flow_y_file = next(gen)
    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
    # Assign an image i to the jth stack in the kth position, but also
    # in the j+1th stack in the k+1th position and so on
    # (for sliding window)
    for s in list(reversed(range(min(L, i + 1)))):
        if i - s < nb_stacks:
            flow[:, :, 2 * s, i - s] = img_x
            flow[:, :, 2 * s + 1, i - s] = img_y
    del img_x, img_y
    gc.collect()
flow = flow - np.tile(flow_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))
flow = np.transpose(flow, [3, 0, 1, 2])
#predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
predictions = []
# Process each stack: do the feed-forward pass
for i in range(flow.shape[0]):
    prediction = model.predict(np.expand_dims(flow[i, ...], 0))
    # predictions[i, ...] = prediction
    predictions.append(prediction)

classifier = load_model("classifier_100")
for prediction in predictions:
    classifier_prediction = classifier.predict(prediction)
    print(classifier_prediction)
