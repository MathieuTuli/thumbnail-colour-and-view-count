import random
import numpy as np
import tensorflow as tf
import math
from math import sqrt
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import colorsys

sess = tf.Session()
red_min = 0
red_max = 15
red_min_2 = 350
red_max_2 = 360
pink_min = 300
pink_max = 350
orange_min = 15
orange_max = 45
yellow_min = 45
yellow_max = 75
green_min = 75
green_max = 155
cyan_min = 155
cyan_max = 200
blue_min = 200
blue_max = 255
purple_min = 255
purple_max = 300
# white_min =
# white_max =
# black_min =
# black_max =

def map_pixels(h,s,v):
    #black
    if(v <= 255):
        return 0,0,0
    #white
    if(s <= 255):
        return 255,255,255
    #red
    if(h >= red_min and h <= red_max):
        return 255, 0, 0
    elif(h >= red_min_2 and h <= red_max_2):
        return 255, 0, 0
    #pink
    elif(h >= pink_min and h <= pink_max):
        return 255, 0, 255
    #orange
    elif(h >= orange_min and h <= orange_max):
        return 255, 125, 0
    #yellow
    elif(h >= yellow_min and h <= yellow_max):
        return 255, 255, 0
    #green
    elif(h >= green_min and h <= green_max):
        return 0, 255, 0
    #cyan
    elif(h >= cyan_min and h <= cyan_max):
        return 0, 255, 255
    #blue
    elif(h >= blue_min and h <= blue_max):
        return 0, 0, 255
    #purple
    elif(h >= purple_min and h <= purple_max):
        return 125, 0, 255

#of all images in jpg_images, return a matrix, size [N,D], D=3
#also return K clusters, randomly assigned to points (rows) in prev matrix
#path is the path to the image directory
#imageType is of format '.XX'
def get_data(K, path, imageType):
    print('Getting data...')
    #the first 2 thumbnails have only been uploaded recently, hence discard
    #them as anomolies
    num_pixels = 0
    numFails = 0
    numSuccesses = 0
    errorCode = 0
    data = []
    clusters = []
    #images in rgb, or 3 colours, 3 columns
    # while(numFails < 10):
    #     try:
    #         im = Image.open(path + str(i) + imageType)
    #         errorCode = 1
    #         rgb_im = im.convert('RGB')
    #         errorCode = 2
    #         r, g, b = rgb_im.getpixel((1, 1))
    #         errorCode = 3
    #         data.append([r,g,b])
    #         errorCode = 4
    #         numSuccesses += 1
    #     except:
    #         numFails += 1
        # i += 1

    img = cv2.cvtColor(cv2.imread('jpg_images/18.jpg'), cv2.COLOR_BGR2HSV)
    w, h, channels = img.shape
    hRatio = 360/179 #cv2 hue is in range 0-179
    svRatio = 1000/255 #s and v in range 0-255
    for i in range(w):
        for j in range(h):
            try:
                h,s,v = img[i,j]
                r,g,b = map_pixels(h * hRatio, s * svRatio, v*svRatio)
                data.append([r,g,b])
                num_pixels += 1
            except:
                continue

    print('Total successful images parsed: ', numSuccesses)
    if errorCode == 0:
        print('ERROR: Couldn\'t open image')
    elif errorCode == 1:
        print('ERROR: Couldn\'t convert to rgb')
    elif errorCode == 2:
        print('ERROR: Couldn\'t get pixel values')
    elif errorCode == 3:
        print('ERROR: Couldn\'t append data')

    for k in range(K):
        try:
            clusters.append(data[random.randint(0, num_pixels)])
            errorCode = 5
        except:
            errorCode = 6
            break
    if errorCode == 4:
        print('ERROR: Couldn\'t append to clusters')

    if errorCode == 5:
        print('Data and clusters successfuly recorded')

    #data is [N1,D] & cluster is [K, D]
    return data, clusters

def show_colors(colors):
    # initialize the bar chart
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    # loop over each cluster and the color of each cluster
    for color in colors:
		# plot the relative percentage of each cluster
        endX = startX + (300/3)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    plt.figure()
    plt.imshow(bar)
    plt.show()
    return
#Takes two vectors (or points) X, Z of size D (D=3)
def update_clusters(data, clusters, K):

    DataExpanded = tf.expand_dims(data,2) # shape [N1, D, 1]
    ClustersExpanded = tf.expand_dims(tf.transpose(clusters),0) # shape [1, D, N2]
    distances = tf.reduce_sum((DataExpanded-ClustersExpanded)**2, 1) # shape [N1,N2]
    assignments = tf.argmin(distances, 1)

    means = []

    for c in range(K):
        means.append(tf.reduce_mean(
            tf.gather(data,
                tf.reshape(
                    tf.where(
                        tf.equal(assignments, c)
                            ), [1,-1])
                                ),reduction_indices=[1]))

    #return of size [K,3]
    return tf.concat(means, 0)

if __name__ == '__main__':
    print('\n\n')
    #define our placeholders
    data = tf.placeholder(tf.float32, name = "data")
    clusters = tf.placeholder(tf.float32, name = "clusters")

    K = 3
    numIterations = 50
    numNearestClusters = 1
    rawData, rawClusters = get_data(K, 'jpg_images/', '.jpg')

    print('Updating clusters...')
    errorCode = 0
    for i in range(numIterations):
        rawClusters = sess.run(update_clusters(data, clusters, K),\
        feed_dict={data:rawData, clusters:rawClusters})

    print(rawClusters)
    show_colors(rawClusters)
