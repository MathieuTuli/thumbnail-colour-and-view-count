
import random
import numpy as np
import tensorflow as tf
from math import sqrt

#define points and clusters
Point = namedtuple('Point', ('coords', 'n', 'count'))
Clusters = namedtuple('Clusters', ('points', 'center', 'n'))

#from a passed in PIL image, create array of points
def get_points(image):
    points = []
    w, h = image.size
    for count, color in image.getcolors(w * h):
        points.append(Point(color, K, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def get_dom_colors(K=3, image):
if __name__ == '__main__':
    get_dom_colors();
