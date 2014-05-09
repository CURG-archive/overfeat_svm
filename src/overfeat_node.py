#!/usr/bin/env python

import numpy as np, pylab as pl, rospy, overfeat, os, cPickle, cv2
from copy import copy, deepcopy
from collections import namedtuple
from functools import partial
from datetime import datetime
from multiprocessing import Process, Pool

from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from scipy.misc import imresize, imsave
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from skimage.feature import hog

OVERFEAT_DIM = 231
KINECT_WIDTH  = 640
KINECT_HEIGHT = 480

# initialize overfeat. 0 = fast net, 1 = accurate net.
overfeat.init("../data/weights/net_weight_0", 0)

# list of unique classes as strings. 
# indices correspond to the 'target' of each 'classification'.
classes  = []

# list of training samples.
training = []

# each Sample is features data and the target of a classification.
Sample = namedtuple('Sample', ['features','depth','classification','time','image'])

# will train svm classifier on features extracted using overfeat
svm_clf = svm.LinearSVC()

# use `image` and `depth` in other functions to access kinect frames.
image = None
depth = None

def listen():
    rospy.init_node("overfeat_preprocessor_node")

    def image_stream(color_image): 
        '''perpetually sets the global `image` to what kinect sees.'''
        global image
        image = color_image
    rospy.Subscriber("/camera/rgb/image_color", numpy_msg(Image), image_stream)

    def depth_stream(depth_image): 
        '''perpetually sets the `depth` to current kinect depth image.'''
        global depth
        depth = depth_image
    rospy.Subscriber("/camera/depth/image/", Image, depth_stream)

def train_svm(include_depth=False, optimize=False):
    '''
    train svm on training Samples. 
    sklearn doesn't have online learning so we have 
    to retrain on the entire data set.
    '''

    X = [np.array(sample.features) for sample in training]
    y = np.array([classes.index(sample.classification) for sample in training])

    if include_depth:
        pool = Pool()
        depths = pool.map(depth_features, [s.depth for s in training])
        pool.close()
        pool.join()

    folds = 10
    skfold = StratifiedKFold(y, folds)

    if optimize:
        optimize_svm_penalty(X,y,skfold)
    else:
        # previously calculated optimal penalty
        svm_clf.C = 6.75

    cv_score = np.mean(cross_val_score(svm_clf, X, y, cv=skfold, n_jobs=-1))
    print 'cv score (%s folds): ' % folds, cv_score

    svm_clf.fit(X,y)

def depth_features(depth_img):
    '''
    fill in np.nan holes using cv2.inpaint,
    then return histogram of oriented gradients
    '''
    start = datetime.now()
    mask   = np.isnan(depth_img).astype(np.uint8)
    normed = (255.0 * depth_img / np.nanmax(depth_img)).astype(np.uint8) 
    filled = cv2.inpaint(normed, mask, 8, cv2.INPAINT_NS)
    hogged = hog(filled, pixels_per_cell=(40,40), cells_per_block=(2,2))
    print 'runtime depth_features(): ', (datetime.now() - start).total_seconds()
    return hogged

Cs = np.linspace(.01, 7, 30)
scores = []
scores_std = []
def optimize_svm_penalty(X, y, cv):
    '''
    set svm penalty param (C) to optimal value using k-fold cv
    '''
    print 'svm optimization started'

    global Cs
    global scores
    global scores_std

    start = datetime.now()
    for C in Cs:
        svm_clf.C = C
        scores_c = cross_val_score(svm_clf, X, y, cv=cv, n_jobs=-1)
        scores.append(np.mean(scores_c))
        scores_std.append(np.std(scores_c))
    print 'runtime %s-fold validation: ' % cv.k, (datetime.now() - start).total_seconds()

    # set optimal svm penalty
    svm_clf.C = Cs[scores.index(max(scores))]
    print 'optimal svm penalty (C): ', svm_clf.C

    # plot k-fold crossvalidation results
    pl.figure(1, figsize=(4, 3))
    pl.clf()
    pl.plot(Cs, scores)
    pl.plot(Cs, np.array(scores) + np.array(scores_std), 'b--')
    pl.plot(Cs, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = pl.yticks()
    pl.yticks(locs, map(lambda x: "%g" % x, locs))
    pl.ylabel('CV score')
    pl.xlabel('Parameter C')
    pl.ylim(0, 1.1)
    pl.show()

def get_and_create_classification_dir(classification):
    '''verifies dir containing classification exists and return formatted dir string'''
    classification_dir = os.path.join("../data/training/", classification)
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir)
    return os.path.abspath(classification_dir)

def add_training_sample(sample):
    if sample.time in [s.time for s in training]:
        # skip adding sample if sample with same timestamp is already in training
        return

    if sample.classification not in classes:
        # if new classification, add to classes
        classes.append(sample.classification)

    training.append(sample)

'''
SCANNING (with the KINECT)

scan functions store data of the current kinect perspective in `scans`.
when you have finished scanning, run save_scans() to persist the data 
to the filesystem as .png and .pickle files. load_scans() will load all 
training data in the /training/data/<classification>/ folders into 
the `training` list, at which point you can run train_svm().
'''
scans = []

def continuously_scan(classification):
    print "classifying: ", classification
    while True:
        key = raw_input("scan? (y/n): ").lower()
        hit_enter = not key
        if key == "y" or hit_enter:
            scan(classification)
        elif key == "n":
            return

def scan(classification):
    '''
    adds the current kinect frame features to our training Samples.
    also saves the image and feature data if specified.
    '''
    time = datetime.now()

    # get image and depth
    decoded_image = np.fromstring(image.data, np.uint8)
    decoded_depth = np.fromstring(depth.data, dtype=np.float32)

    scans.append({
        'classification': classification,
        'depth': decoded_depth,
        'image': decoded_image,
        'time': time,
    })

def save_scans():
    '''
    save all scans to .png and .pickle files.
    job runs in the background.
    each scan gets its own pickle (and image, of course).
    '''

    def save_scan(classification, image, depth, time):
        '''
        save individual scan to png and pickle. the pickle is merely a 
        dump of the Sample namedtuple. overfeat feature extaction is 
        performed before creating the Sample and therefore before pickling.
        '''
        # reshape image and depth 
        reshaped_img   = np.reshape(image, (KINECT_HEIGHT, KINECT_WIDTH, 3))
        reshaped_depth = np.reshape(depth, (KINECT_HEIGHT, KINECT_WIDTH, 1))

        # resize and rearrange image RGB order for overfeat
        resized    = imresize(reshaped_img, (OVERFEAT_DIM, OVERFEAT_DIM)).astype(np.float32)
        flattened  = resized.reshape(OVERFEAT_DIM*OVERFEAT_DIM, 3)
        rearranged = flattened.transpose().reshape(3, OVERFEAT_DIM, OVERFEAT_DIM)

        # extract image features via overfeat
        _, features = run_overfeat(rearranged)

        # save resized image
        filename = time.strftime("%Y-%m-%d-%H-%M-%S-%f")
        save_image(resized, classification, filename)

        # create sample and write it to file
        sample = Sample(features, reshaped_depth, classification, time, reshaped_img)
        save_sample(sample, filename)

    def save_image(img, classification, filename):
        '''saves a *numpy array* as an png to ../data/training/<classification>/<filename>.png'''
        classification_dir = get_and_create_classification_dir(classification)
        filename = filename + ".png"
        image_path = os.path.join(classification_dir, filename)
        imsave(image_path, img)

    def save_sample(sample, filename):
        '''pickles a sample to ../data/training/<classification>/<filename>.pickle'''
        classification_dir = get_and_create_classification_dir(sample.classification)
        filename = filename + ".pickle"
        pickle_path = os.path.join(classification_dir, filename)
        with open(pickle_path, 'w') as outfile:
            cPickle.dump(sample, outfile)

    # copy and reset `scans` so we can continue scanning
    global scans
    scans_copy = deepcopy(scans)
    scans = []

    # run save_scans() on each scan in background process
    def run_save_scans(scans):
        for scan in scans_copy:
            save_scan(**scan)
    p = Process(
        target = run_save_scans,
        args = (scans_copy,)),
    p.start()

def load_scans():
    '''
    loads .pickled samples from ../data/training/
    note: this is *slow*. 2 mins. write entire loaded training
    list to single pickle and unpickle from that single file.

    TODO add classes. restrict loading to certain classes.
    TODO add datetime range. since filenames are simply datetimes,
         load filename into datatime and check to make sure in range.
    '''
    start = datetime.now()

    # get all pickle files
    training_dir = os.path.abspath("../data/training")
    for classification in os.listdir(training_dir):
        classification_path = os.path.join(training_dir, classification)
        for filename in os.listdir(classification_path):
            _, extension = os.path.splitext(filename)
            if extension == ".pickle":
                pickle_path = os.path.join(classification_path, filename)
                with open(pickle_path, 'r') as pickle_file:
                    sample = cPickle.load(pickle_file)
                    add_training_sample(sample)

    print 'runtime load_scans(): ', (datetime.now() - start).total_seconds()

    '''
    def unpickle(pickle_path):
        """expensive depickling in own fcn to be done in parallel"""
        with open(pickle_path, 'r') as pickle_file:
            sample = cPickle.load(pickle_file)
            return sample
    
    # concurrently unpickle training data
    pool = Pool(8)
    unpickled = pool.map(unpickle, pickle_paths)
    pool.close()
    pool.join()

    # properly handle adding samples to global training list
    for sample in unpickled:
        add_training_sample(sample)
    '''

def predict(method="svm", n=1):
    ''' 
    classify current image using a specified method
    method
        "svm" - classify using svm trained on overfeat extracted features
        "overfeat" - classify current image using overfeat
    n
        returns multiple guesses. only works for method="overfeat"
    '''
    # resize image and rearrange RGB order for overfeat
    decoded = np.fromstring(image.data, np.uint8)
    reshaped = np.reshape(decoded, (image.height, image.width, 3))
    resized = imresize(reshaped, (OVERFEAT_DIM,OVERFEAT_DIM)).astype(np.float32)
    flattened  = resized.reshape(OVERFEAT_DIM*OVERFEAT_DIM, 3)
    rearranged = flattened.transpose().reshape(3, OVERFEAT_DIM, OVERFEAT_DIM)

    # extract features
    likelihoods, features = run_overfeat(rearranged)

    if method == "overfeat":
        for prediction in overfeat_predictions(likelihoods, n):
            print prediction
    elif method == "svm":
        target = int(svm_clf.predict([features])[0])
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
    formatted_features    = copy(overfeat_features.flatten())
    formatted_likelihoods = copy(overfeat_likelihoods.flatten())
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
    TODO FIX THIS SHIT
    prepare an image for overfeat.

    save_func function params: 
        image: image to be saved.

    save_func should already know what filename to save as.
    '''
    # decode image data from string
    # decoded  = np.fromstring(image.data, np.uint8)
    reshaped = np.reshape(decoded, (image.height, image.width, 3))

    # resize image for overfeat
    dim = 231
    resized = imresize(reshaped, (dim,dim)).astype(np.float32)

    if save_func:
        save_func(resized)

    # rearrange RGB order
    flattened  = resized.reshape(OVERFEAT_DIM*OVERFEAT_DIM, 3)
    rearranged = flattened.transpose().reshape(3, OVERFEAT_DIM, OVERFEAT_DIM)
    return rearranged
