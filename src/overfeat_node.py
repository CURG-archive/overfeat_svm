#!/usr/bin/env python

import numpy as np, rospy, overfeat, os, pickle
from copy import copy
from collections import namedtuple
from functools import partial
from datetime import datetime

from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from scipy.misc import imresize, imsave
from sklearn import svm

# initialize overfeat. 0 = fast net, 1 = accurate net.
overfeat.init("../data/weights/net_weight_0", 0)

# list of unique classes as strings. 
# indices correspond to the 'target' of each 'classification'.
classes  = []

# list of training samples.
# each Sample is features data and the target of a classification.
Sample = namedtuple('Sample', ['data','classification','time'])
training = []

# will train svm classifier on features extracted using overfeat
classifier = svm.SVC()

# use `image` in other functions to access kinect frames.
global image
image = None

def listen():
    rospy.init_node("overfeat_preprocessor_node")

    def image_stream(img): 
        '''
        perpetually sets the `image` to current kinect perspective. 
        use `image` in other functions to access kinect frames.
        '''
        global image
        image = img

    # processes kinect stream through image_stream()
    rospy.Subscriber("/camera/rgb/image_color", numpy_msg(Image), image_stream)

def train_svm():
    '''
    train svm on training Samples. 
    sklearn doesn't have online learning so we have 
    to retrain on the entire data set.
    '''
    data = [np.array(sample.data) for sample in training]
    targets = np.array([classes.index(sample.classification) for sample in training])
    classifier.fit(data, targets)

def get_and_create_classification_dir(classification):
    '''verifies dir containing classification exists and return formatted dir string'''
    classification_dir = os.path.join("../data/training/", classification)
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir)
    return os.path.abspath(classification_dir)

def save_image(img, classification, filename):
    '''
    saves a *numpy array* as an image in ../data/training/<classification>/<filename>.<suffix>.png.
    TODO change from saving a numpy array to something more robust. currently just 
         using this fcn for viewing by hand rather than storing for reprocessing.

    classification
        the string describing the object
    '''
    classification_dir = get_and_create_classification_dir(classification)
    filename = filename + ".png"
    image_path = os.path.join(classification_dir, filename)
    imsave(image_path, img)

def save_sample(sample, filename):
    classification_dir = get_and_create_classification_dir(sample.classification)
    filename = filename + ".pickle"
    pickle_path = os.path.join(classification_dir, filename)
    with open(pickle_path, 'w') as outfile:
        pickle.dump(sample, outfile)

def load_samples():
    '''
    loads .pickled samples from ../data/training/

    TODO add classes. restrict loading to certain classes.
    TODO add datetime range. since filenames are simply datetimes,
         load filename into datatime and check to make sure in range.
    '''
    training_dir = os.abspath("../data/training")
    for classification in os.listdir(training_dir):
        classification_path = os.path.join(training_dir, classification)
        for filename in os.listdir(classification_path):
            _, extension = os.path.splitext(filename)
            if extension == ".pickle":
                pickle_path = os.path.join(classification_path, filename)
                with open(pickle_path, 'r') as pickle_file:
                    sample = pickle.load(pickle_file)
                    add_training_sample(sample)

def save_depth():
    '''TODO maybe better to incorporate this functionality into save_image()'''
    pass

def add_training_sample(sample):
    if sample.classification not in classes:
        # if new classification, add to classes
        classes.append(sample.classification)
    training.append(sample)

def continuously_save_training_data(classification):
    print "classifying: ", classification
    while True:
        in = raw_input("scan? (y/n): ").lower()
        if in == "y":
            save_training_data(classification)
        elif in == "n":
            return
        else:
            print "input y or n"
            continue

def save_training_data(classification, do_save_image=True, do_save_sample=True):
    '''
    adds the current kinect frame features to our training Samples.
    also saves the image and feature data if specified.
    '''
    time = datetime.now()
    filename = time.strftime("%Y-%m-%d-%H-%M-%S-%f")

    # partially apply our save image fcn to be used inside of preprocess()
    save_processed_image_func = partial(
        save_image, 
        classification = classification,
        filename = "%s.%s" % (filename, "processed"),
    ) if do_save_image else None

    # extract image features via overfeat and write to file if specified
    global image
    processed = preprocess(image, save_processed_image_func)
    _, features = run_overfeat(processed)

    # create sample, add it to training list, write it to file for re-use
    sample = Sample(features, classification, time)
    #add_training_sample(sample)
    if do_save_sample:
        save_sample(sample, filename)

def predict(method="svm", n=1):
    ''' 
    classify current image using a specified method
    method
        "svm" - classify using svm trained on overfeat extracted features
        "overfeat" - classify current image using overfeat
    n
        returns multiple guesses. only works for method="overfeat"
    '''
    processed = preprocess(image)
    feats, likelihoods = run_overfeat(processed)

    if method == "overfeat":
        for prediction in overfeat_predictions(likelihoods, n):
            print prediction
    elif method == "svm":
        target = int(classifier.predict([feats])[0])
        print classes[target]

def run_overfeat(image, layer=None):
    '''
    runs an image through overfeat. returns the 1,000-length likelihoods 
    vector and N-length layer in the net as copied and formatted numpy arrays.
    layer
        None: means return 4,096-length feature vector just prior to the output layer.
        int:  return that layer instead.
    '''
    if not layer:
        # get layer just before output layer
        feature_layer = overfeat.get_n_layers() - 2

    overfeat_likelihoods  = overfeat.fprop(image)
    overfeat_features     = overfeat.get_output(feature_layer)

    # flatten and copy. NOTE: copy() is intentional, don't delete.
    formatted_likelihoods = copy(overfeat_features.flatten())
    formatted_features    = copy(overfeat_likelihoods.flatten())
    return formatted_likelihoods, formatted_features

def overfeat_predictions(likelihoods, n=1):
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

def preprocess(image, save_func=None):
    '''
    prepare an image for overfeat.

    save_func function params: 
        image: image to be saved.

    save_func should already know what filename to save as.
    '''
    # decode image data from string
    decoded  = np.fromstring(image.data, np.uint8)
    reshaped = np.reshape(decoded, (image.height, image.width, 3))

    # resize image for overfeat
    dim = 231
    resized = imresize(reshaped, (dim,dim)).astype(np.float32)

    if save_func:
        save_func(resized)

    # rearrange RGB order
    flattened  = resized.reshape(dim*dim, 3)
    rearranged = flattened.transpose().reshape(3, dim, dim)
    return rearranged
