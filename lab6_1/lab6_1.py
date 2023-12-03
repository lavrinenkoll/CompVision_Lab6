
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from matching import *


# read image from path
def read_image (path):
    image = cv2.imread(path)
    # show image
    plt.imshow(image)
    plt.title('Оригінал зображення')
    plt.show()
    return image


# edit image with filters
def image_edit(image, type):
    # edit image depending on type
    if type == 1:
        # convert image to HLS
        im = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # normalize image
        im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # apply filter to image to make it more clear and bright
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        im = cv2.filter2D(im, -1, kernel)
        # apply blur to image
        im = cv2.GaussianBlur(im, (3, 3), 0)
        #im = cv2.medianBlur(im, 3)

        edited = im.copy()
    else:
        # convert image to HLS
        im = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # normalize image
        im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # apply filter to image to make it more clear and bright
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        im = cv2.filter2D(im, -1, kernel)
        # apply blur to image with different parameters
        im = cv2.GaussianBlur(im, (3, 3), 0)
        im = cv2.blur(im, (8, 8))

        edited = im.copy()

    # show edited image
    plt.imshow(edited)
    plt.title('Зображення після корекції кольору')
    #plt.show()
    return edited


# color clusterization
def color_clusterization(image, type):
    # set number of clusters depending on type
    if type == 1:
        n_clusters = 4
    elif type == 2:
        n_clusters = 3
    # convert image to RGB from HLS
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)

    # reshape image
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))

    # clusterize image
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(pixels).reshape(image.shape[:2])

    # create new image with clusters
    img = image.copy()
    # create new image with only one cluster
    img_new = image.copy()
    # set cluster to find
    if type == 1:
        k = labels[900][100]
    elif type == 2:
        k = labels[900][100]

    # set color for each cluster on img1, if cluster is not needed, set it to white on img2
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == 0:
                img[i][j] = [0, 0, 0]
                if k != 0:
                    img_new[i][j] = [255, 255, 255]
            elif labels[i][j] == 1:
                img[i][j] = [255, 0, 0]
                if k != 1:
                    img_new[i][j] = [255, 255, 255]
            elif labels[i][j] == 2:
                img[i][j] = [0, 255, 0]
                if k != 2:
                    img_new[i][j] = [255, 255, 255]
            elif labels[i][j] == 3:
                img[i][j] = [0, 0, 255]
                if k != 3:
                    img_new[i][j] = [255, 255, 255]
            elif labels[i][j] == 4:
                img[i][j] = [255, 255, 0]
                if k != 4:
                    img_new[i][j] = [255, 255, 255]

    # show images
    plt.imshow(img)
    plt.title('Зображення після кластеризації')
    #plt.show()
    plt.imshow(img_new)
    plt.title('Зображення після виділення кластеру')
    #plt.show()
    return img, img_new


# segmentation with Robert's operator
def segmentation(img):
    # Robert's operator
    roberts_cross_v = np.array([[1, 0], [0, -1]])
    roberts_cross_h = np.array([[0, 1], [-1, 0]])

    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype('float64') / 255.0

    # apply Robert's operator
    vertical = ndimage.convolve(img, roberts_cross_v)
    horizontal = ndimage.convolve(img, roberts_cross_h)

    # gradient transformation
    segment_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    # normalize image
    segment_img *= 255

    # show image
    plt.imshow(segment_img)
    plt.title('Зображення після сегментації')
    #plt.show()
    return segment_img


# find contours in edited image
def image_contour(image):
    image = np.array(image, np.uint8)
    # find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# draw all input contours
def draw_contour(image, contours):
    # draw contours on image with random colors
    for countour in contours:
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        cv2.drawContours(image, countour, -1, (r, g, b), 3)
    # show image with contours
    plt.imshow(image)
    plt.title('Зображення після виділення контурів')
    #plt.show()


# compare contours with geometric object
def image_recognition (image_entrance, image_cont):
    image_all = image_entrance.copy()
    buildings = []
    buildings_cont = []
    water = []
    water_cont = []

    # analyze each contour
    for c in image_cont:
        # find perimeter and approximate contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # create rectangle around contour
        rectangle = cv2.boundingRect(approx)
        # find area of contour
        area = cv2.contourArea(approx)

        # if area is between 100 and 70, contour is house,
        # if area is between 1500 and 70000, contour is water
        if area < 1000 and area > 70:
            cv2.drawContours(image_all, [approx], -1, (255, 0, 0), 2)
            cv2.rectangle(image_all, (rectangle[0], rectangle[1]), (rectangle[0]+rectangle[2], rectangle[1]+rectangle[3]), (255, 0, 0), 2)
            buildings.append(rectangle)
            buildings_cont.append(approx)
            #cv2.putText(image_all,"Будинок", (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
        elif area > 1500 and area < 70000:
            cv2.drawContours(image_all, [approx], -1, (0, 0, 255), 2)
            cv2.rectangle(image_all, (rectangle[0], rectangle[1]), (rectangle[0]+rectangle[2], rectangle[1]+rectangle[3]), (0, 0, 255), 2)
            water.append(rectangle)
            water_cont.append(approx)
            #cv2.putText(image_all,"Водойма", (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))

    # show image with contours
    plt.imshow(image_all)
    plt.title('Зображення після виділення контурів')
    #plt.show()

    return buildings, buildings_cont, water, water_cont


def image_of_contour (img1, cont1, img2, cont2):
    img1_new = cv2.zeros(img1.shape[:2], np.uint8)
    img2_new = cv2.zeros(img2.shape[:2], np.uint8)

    cv2.fillPoly(img1_new, [cont1], (255, 255, 255))
    cv2.fillPoly(img2_new, [cont2], (255, 255, 255))

    plt.imshow(img1_new)
    plt.title('Зображення після виділення контурів')
    #plt.show()




# main
path1 = 'img.png'
path2 = 'img_2.png'

image1 = read_image(path1)
image_edited1 = image_edit(image1, 2)
clusters1, img_new1 = color_clusterization(image_edited1, 2)
edged1 = segmentation(img_new1)
contours1 = image_contour(edged1)
# draw_contour(image, contours)
buildings1, buildings_cont1, water1, water_cont1 = image_recognition(image1, contours1)

image2 = read_image(path2)
image_edited2 = image_edit(image2, 1)
clusters2, img_new2 = color_clusterization(image_edited2, 1)
edged2 = segmentation(img_new2)
contours2 = image_contour(edged2)
# draw_contour(image, contours)
buildings2, buildings_cont2, water2, water_cont2 = image_recognition(image2, contours2)

matching_func('img.png', 'img_2.png', water1, water2)
