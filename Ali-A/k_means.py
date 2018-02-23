import random
import numpy as np
import tensorflow as tf
import math
from math import sqrt
from PIL import Image

sess = tf.Session()

#of all images in jpg_images, return a matrix, size [N,D], D=3
#also return K clusters, randomly assigned to points (rows) in prev matrix
#path is the path to the image directory
#imageType is of format '.XX'
def get_data(K, path, imageType):
    print('Getting data...')
    #the first 2 thumbnails have only been uploaded recently, hence discard
    #them as anomolies
    i = 1
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

    img = Image.open('test_image.jpg')
    img.thumbnail((10,10))
    w, h = img.size
    rgb_im = img.convert('RGB')
    for i in range(w):
        for j in range(h):
            r, g, b = rgb_im.getpixel((i, j))
            data.append([r,g,b])

    print('Total successful images parsed: ', numSuccesses, '.')
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
            clusters.append(data[random.randint(0, i)])
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

#Takes two vectors (or points) X, Z of size D (D=3)
def eucl_dist(X,Z):
    XExpanded = tf.expand_dims(X,2) # shape [N1, D, 1]
    ZExpanded = tf.expand_dims(tf.transpose(Z),0) # shape [1, D, N2]
    return tf.reduce_sum((XExpanded-ZExpanded)**2, 1) # shape [N1,N2]

def nearest_cluster(data, clusters, K):
    clusterDistance = eucl_dist(data, clusters)
    nearestClusters, clusterIndices = tf.nn.top_k(-clusterDistance, k = K)
    return clusterIndices

def update_clusters(data, clusters, clusterIndices, K):
    updatedClusters = tf.to_float(tf.expand_dims(tf.fill(tf.shape(clusters), 0), 0))

    for i in range(clusterIndices.shape[0]):
        currentRow = tf.expand_dims(tf.expand_dims(tf.to_float(tf.gather(data, i)), 0), 0)
        whichCluster = clusterIndices[i]

        #rgb, so 3 cols
        emptyRow = [[[0,0,0]]]
        concatinated = []
        if whichCluster == 0:
            concatinated = tf.concat([currentRow, emptyRow], 1)
            concatinated = tf.concat([concatinated, emptyRow], 1)
        elif whichCluster == 1:
            concatinated = tf.concat([emptyRow, currentRow], 1)
            concatinated = tf.concat([concatinated, emptyRow], 1)
        else:
            concatinated = tf.concat([emptyRow, currentRow], 1)
            concatinated = tf.concat([emptyRow, concatinated], 1)

        updatedClusters = tf.concat([updatedClusters, concatinated], 0)

    #return of size [K,3]
    return tf.reduce_mean(tf.to_float(updatedClusters), 0)

if __name__ == '__main__':
    print('\n\n')
    #define our placeholders
    data = tf.placeholder(tf.float32, name = "data")
    clusters = tf.placeholder(tf.float32, name = "clusters")

    K = 3
    numIterations = 10
    numNearestClusters = 1
    rawData, rawClusters = get_data(K, 'jpg_images/', '.jpg')

    print('Updating clusters...')
    errorCode = 0
    for i in range(numIterations):
        clusterIndices = sess.run(nearest_cluster(data, clusters, numNearestClusters),\
        feed_dict={data:rawData, clusters:rawClusters})

        rawClusters = sess.run(update_clusters(data, clusters, clusterIndices, K),\
        feed_dict={data:rawData, clusters:rawClusters})

    print(rawClusters)
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
#
# points_n = 200
# clusters_n = 3
# iteration_n = 100
#
# points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
# centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
#
# points_expanded = tf.expand_dims(points, 0)
# centroids_expanded = tf.expand_dims(centroids, 1)
#
# distances = tf.reduce_sum(tf.square(points_expanded - centroids_expanded), 2)
# assignments = tf.argmin(distances, 0)
#
# means = []
# for c in range(clusters_n):
#     means.append(tf.reduce_mean(
#       tf.gather(points,
#                 tf.reshape(
#                   tf.where(
#                     tf.equal(assignments, c)
#                   ),[1,-1])
#                ),reduction_indices=[1]))
#
# new_centroids = tf.concat(means,0)
#
# update_centroids = tf.assign(centroids, new_centroids)
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#   sess.run(init)
#   for step in range(iteration_n):
#     [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])
#
#   print ("centroids" + "\n", centroid_values)
#
# plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
# plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
# plt.show()
