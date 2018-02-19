# def most_frequent_colour(image):
#     w, h = image.size
#
#     #returns a tuple. the colours, and their frequency
#     pixels = image.getcolors(w * h)
#
#     #set to first one. will iterate to find the most frequent later
#     most_frequent_pixel = pixels[0]
#
#     #find the most frequent colour
#     for count, colour in pixels:
#         if count > most_frequent_pixel[0]:
#             most_frequent_pixel = (count, colour)
#
#     print(most_frequent_pixel[1])
#     image.show()
#     plt.imshow([most_frequent_pixel[1]])
#     plt.show()
#
#     return most_frequent_pixel
#
# def eucl_dist(X,Z):
#     XExpanded = tf.expand_dims(X,2) # shape [N1, D, 1]
#     ZExpanded = tf.expand_dims(tf.transpose(Z),0) # shape [1, D, N2]
#     #for both...axis2 = D. for axis0 and axis2, there is a corresponding size 1.
#     #makes them compatible for broadcasting
#
#     #return the reduced sum accross axis 1. This will sum accros the D dimensional
#     #element thus returning the N1xN2 matrix we desire
#     return tf.reduce_sum((XExpanded-ZExpanded)**2, 1)
#
# def eucl_dist2(p1, p2):
#     return sqrt(sum([
#         (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
#     ]))
#
# #image is of type Image.open('filename')
# def get_points(image):
#     points = []
#     w, h = image.size
#     for count, color in image.getcolors(w * h):
#         points.append(Point(color, 3, count))
#     return points
#
# #image is of type Image.open('filename')
# def colours(image, n=3):
#     image.thumbnail((200,200))
#     w, h = image.size
#
#     points = get_points(image)
#     clusters = kmeans(points, n, 1)
#     rgbs = [map(int, c.center.coords) for c in clusters]
#
#     return map(rtoh, rgbs)
#
# def calculate_center(points, n):
#     vals = [0.0 for i in range(n)]
#     plen = 0
#
#     for p in points:
#         plen += p.ct
#         for i in range(n):
#             vals[i] += (p.coords[i] * p.ct)
#
#     return Point([(v / plen) for v in vals], n, 1)
#
# def kmeans(points, k, mind_diff):
#     cluster = [Cluster([p], p, p.n) for p in random.sample(points, k)]
#
#     while True:
#         plists = [[] for i in range(k)]
#         for p in points:
#             smallest_distance = float('Inf')
#             for i in range(k):
#                 distance = eucl_dist2(p, cluster[i].center)
#                 if distance < smallest_distance:
#                     smallest_distance = distance
#                     idx = i
#             plist[idx].append(p)
#
#         diff = 0
#         for i in range(k):
#             old = clusters[i]
#             center = calculate_center(plists[i], old.n)
#             new = Cluster(plists[i], center, old.n)
#             clusters[i] = new
#             diff = max(diff, eucl_dist2(old.center, new.center))
#
#         if diff < min.diff:
#             break
#
#     return clusters

from collections import namedtuple
from math import sqrt
import random
try:
    import Image
except ImportError:
    from PIL import Image

Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

def get_points(img):
    points = []
    w, h = img.size
    for count, color in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(filename, n=3):
    img = Image.open(filename)
    # img.thumbnail((200, 200))
    w, h = img.size

    points = get_points(img)
    clusters = kmeans(points, n, 1)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([
        (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
    ]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters

if __name__ == '__main__':
    returned = colorz('jpg_images/0.jpg')
    list(map(print, returned))
