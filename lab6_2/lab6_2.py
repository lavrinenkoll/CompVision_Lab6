import math

import cv2
import keras.utils as image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os, shutil, glob, os.path


def clustering_with_a_teacher (path, p):
    image.LOAD_TRUNCATED_IMAGES = True
    model = ResNet50(weights='imagenet', include_top=False)

    imdir = path
    targetdir = p
    number_clusters = 7

    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, '*.png'))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" % (i, len(filelist)), end="\r")
        img = image.load_img(imagepath, target_size=(224, 224)) # VGG must take 224x224 images as input
        img_data = image.img_to_array(img) # Convert to Numpy array with shape (224,224,3)
        img_data = np.expand_dims(img_data, axis=0) # Convert Numpy array with shape (224,224,3) to (1,224,224,3)
        img_data = preprocess_input(img_data) # Apply preprocessing for model input
        features = np.array(model.predict(img_data)) # Extract features using VGG16
        featurelist.append(features.flatten()) # Add to feature list

    # Clustering
    kmeans = KMeans(n_clusters=number_clusters, init='k-means++', n_init=10, random_state=0).fit(np.array(featurelist))

    # Copy images renamed by cluster
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    dict = {}
    for i in range(number_clusters):
        dict[i] = []

    for i, m in enumerate(kmeans.labels_):
        print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
        shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
        dict[m].append(cv2.imread(filelist[i]))


    for i in range(number_clusters):
        lines = round(math.sqrt(len(dict[i])))
        img_line = []
        for j in range(lines):
            images = 0
            for k in range(images+j*lines, images+j*lines+lines):
                try:
                    dict[i][k]
                except:
                    break
                if images == 0:
                    img = dict[i][k]
                else:
                    if img.shape[0] < dict[i][k].shape[0]:
                        img = np.concatenate((img, np.zeros((dict[i][k].shape[0] - img.shape[0], img.shape[1], 3))), axis=0)
                    elif img.shape[0] > dict[i][k].shape[0]:
                        dict[i][k] = np.concatenate((dict[i][k], np.zeros((img.shape[0] - dict[i][k].shape[0], dict[i][k].shape[1], 3))), axis=0)

                    img = np.concatenate((img, dict[i][k]), axis=1)
                images += 1
            img_line.append(img)
        img = img_line[0]
        for j in range(1, len(img_line)):
            if img.shape[1] < img_line[j].shape[1]:
                img = np.concatenate((img, np.zeros((img.shape[0], img_line[j].shape[1] - img.shape[1], 3))), axis=1)
            elif img.shape[1] > img_line[j].shape[1]:
                img_line[j] = np.concatenate((img_line[j], np.zeros((img_line[j].shape[0], img.shape[1] - img_line[j].shape[1], 3))), axis=1)
            img = np.concatenate((img, img_line[j]), axis=0)
        img = img / 255.0
        plt.imshow(img)
        plt.show()

clustering_with_a_teacher('C:\Workspace/7semestr\compvision\lab6\lab6_2\Starter_set',
                          'C:\Workspace/7semestr\compvision\lab6\lab6_2\Final_set/')