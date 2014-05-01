#!/usr/bin/env python

import numpy as np, rospy, overfeat
from copy import copy
from collections import namedtuple

from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from scipy.misc import imresize
from sklearn import svm

# initialize overfeat
overfeat.init("../data/default/net_weight_0", 0)

# list of training data and their classifications
training = []
targets  = []

# will train svm classifier on features extracted using overfeat
classifier = svm.SVC()

# store current image from stream of images coming from the kinect
global image
image = None

def listen():
    rospy.init_node("overfeat_preprocessor_node")

    def image_stream(img): 
        ''' handler for processing stream of images coming from the kinect '''
        global image
        image = img

    rospy.Subscriber(
        "/camera/rgb/image_color",
        numpy_msg(Image),
        image_stream)

def train_svm():
    '''
    train svm on entire data set. 
    sklearn doesn't have online learning so we have 
    to retrain on the entire data set.
    '''
    classifier.fit(np.array(training), np.array(targets))

def add_training_image(classification=0):
    global image
    processed = preprocess(image)
    _, feats = run_overfeat(processed)
    training.append(np.array(feats))
    targets.append(classification)

def predict(method="svm", n=1):
    ''' 
    classify current image using a specified method
    method
        "svm" - classify using svm trained on overfeat extracted features
        "overfeat" - classify current image using overfeat
    '''
    processed = preprocess(image)
    likelihoods, feats = run_overfeat(processed)

    if method == "overfeat":
        for prediction in top_n_predictions(likelihoods, n):
            print prediction
    elif method == "svm":
        print classifier.predict([feats])[0]

def run_overfeat(image, feature_layer=None):
    if not feature_layer:
        feature_layer = overfeat.get_n_layers() - 2

    likelihoods = copy(overfeat.fprop(image).flatten())
    features    = copy(overfeat.get_output(feature_layer).flatten())
    return likelihoods, features

def top_n_predictions(likelihoods, n=1):
    '''
    returns top n predictions given an overfeat likelihoods 
    vector whose index corresponds with a category
    '''
    assert len(likelihoods) == 1000
    assert n >= 1 and n <= 1000

    Prediction = namedtuple('Prediction', ['name_index','likelihood'])
    predictions = (Prediction(i,v) for i,v in enumerate(likelihoods))

    # sort prediction by descending likelihood 
    predictions = sorted(predictions, key=lambda x: -x.likelihood)

    return [overfeat.get_class_name(pred.name_index) for pred in predictions[0:n]]

def preprocess(image):
    '''prepare an image for overfeat'''
    # decode image data from string
    decoded  = np.fromstring(image.data, np.uint8)
    reshaped = np.reshape(decoded, (image.height, image.width, 3))

    # resize image for overfeat
    dim = 231
    resized = imresize(reshaped, (dim,dim)).astype(np.float32)

    # rearrange RGB order
    flattened  = resized.reshape(dim*dim, 3)
    rearranged = flattened.transpose().reshape(3, dim, dim)
    return rearranged
