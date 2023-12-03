import math
import os
import shutil
import cv2
import keras.utils as image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def clustering_img(path_input, path_output, number_clusters):
    # set allow to load truncated images
    image.LOAD_TRUNCATED_IMAGES = True

    # load model ResNet50 with weights from ImageNet, without top layer
    model = ResNet50(weights='imagenet', include_top=False)

    # load images from path_input
    paths_images = [os.path.join(path_input, file) for file in os.listdir(path_input) if file.endswith('.png')]
    paths_images.sort()

    features_of_img = []
    # load features for each image
    for i in range(len(paths_images)):
        # load image from path and resize it to 224x224, because ResNet50 must take 224x224 images as input
        img = image.load_img(paths_images[i], target_size=(224, 224))
        # convert image to HLS color space for better clustering
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HLS)
        # convert image to array with shape (224,224,3)
        img_data = image.img_to_array(img)
        # convert image to array with shape (1,224,224,3), because ResNet50 must take 4D array as input
        img_data = img_data.reshape((1,) + img_data.shape)
        # apply preprocessing for model input, normalize image by subtracting mean RGB values from ImageNet
        img_data = preprocess_input(img_data)
        # extract features using ResNet50
        features = np.array(model.predict(img_data))
        # save features, flatten array to 1D
        features_of_img.append(features.flatten())

    # clustering images, use KMeans algorithm
    kmeans = (KMeans(n_clusters=number_clusters, init='k-means++', n_init=10, random_state=0)
              .fit(np.array(features_of_img)))

    # create output directory
    try:
        os.makedirs(path_output)
    except OSError:
        # if directory exists, remove it and create new one
        shutil.rmtree(path_output)
        os.makedirs(path_output)

    # create dictionary for images in each cluster
    dict_img = {}
    for i in range(number_clusters):
        # create key for dictionary for each cluster
        dict_img[i] = []

    # save images to output directory, rename images by cluster
    for i in range(len(kmeans.labels_)):
        shutil.copy(paths_images[i], path_output + str(kmeans.labels_[i]+1) + "_" + paths_images[i].split('\\')[-1])
        # read image
        dict_img[kmeans.labels_[i]].append(cv2.imread(paths_images[i]))


    # show result for each cluster in separate plot
    for i in range(number_clusters):
        # calculate number of images in line
        lines = round(math.sqrt(len(dict_img[i])))
        # create list for images in line
        img_line = []
        # load images from cluster
        for j in range(lines):
            # calculate number of loaded images
            images = 0
            # load images in line from last image in line to first
            for k in range(images+j*lines, images+j*lines+lines):
                # check if image exists
                try:
                    dict_img[i][k]
                except:
                    break
                # if it is first image in line, save it to img
                if images == 0:
                    img = dict_img[i][k]
                else:
                    # if image in line has different height, add zeros to image with smaller height
                    if img.shape[0] < dict_img[i][k].shape[0]:
                        img = np.concatenate((img, np.zeros((dict_img[i][k].shape[0] - img.shape[0], img.shape[1], 3))), axis=0)
                    elif img.shape[0] > dict_img[i][k].shape[0]:
                        dict_img[i][k] = np.concatenate((dict_img[i][k], np.zeros((img.shape[0] - dict_img[i][k].shape[0], dict_img[i][k].shape[1], 3))), axis=0)
                    # concatenate images in line
                    img = np.concatenate((img, dict_img[i][k]), axis=1)
                # increase number of loaded images
                images += 1
            # add line of images to list of lines
            img_line.append(img)

        # concatenate lines of images
        img = img_line[0]
        for j in range(1, len(img_line)):
            # if image in line has different width, add zeros to image with smaller width
            if img.shape[1] < img_line[j].shape[1]:
                img = np.concatenate((img, np.zeros((img.shape[0], img_line[j].shape[1] - img.shape[1], 3))), axis=1)
            elif img.shape[1] > img_line[j].shape[1]:
                img_line[j] = np.concatenate((img_line[j], np.zeros((img_line[j].shape[0], img.shape[1] - img_line[j].shape[1], 3))), axis=1)
            # concatenate lines of images
            img = np.concatenate((img, img_line[j]), axis=0)
        # normalize image
        img = img / 255.0
        # show image
        plt.imshow(img)
        plt.title('Cluster ' + str(i+1))
        plt.show()


clustering_img('Starter_set', 'Final_set/', 7)
