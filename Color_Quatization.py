### Task 1: Color Quantization
# img03 - 960*400
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import math
import random
import argparse
import os
import cv2
import time

def Random_Centers(img, k):
    im = img
    img = array(im)
    # np.random.seed(42)
    random_centers = []
    m = np.shape(img)[0]
    n = np.shape(img)[1]

    for _ in range(k):
        r_col = math.floor(np.random.uniform(0, m, 1))
        r_row = math.floor(np.random.uniform(0, n, 1))
        random_centers.append(img[r_col, r_row])
    return random_centers

def Manual_Centers(img, k):
    manual_centers = []
    # pick k initial color centers manually by clicking on the image
    plt.imshow(img)
    centers = plt.ginput(k, show_clicks=True)
    for i in centers:
        ii = list(i)
        [col, row] = ii
        f_col = math.floor(col)
        f_row = math.floor(row)
        pixel = img[(f_row, f_col)]
        manual_centers.append(pixel)
    return manual_centers

def ClusteringPixels(img, cluster_centers):
    """
    Computes distance of each pixel to the randomly or manually chosen cluster centers
    and then assign pixels to the closest cluster

    :param img: location of the image that will be used
    :param cluster_centers: Initial cluster centers (manually or randomly selected)
    :return: dictionary of pixel width and hight and no of the belonged clusters
    """
    im = img
    img = array(im)
    class_vector = {}
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            pixel = img[row, col]
            distance = np.linalg.norm(np.uint8([pixel]) - cluster_centers, axis=1)
            class_vector[(row, col)] = np.argmin(distance)
    return class_vector

def updateClusterCenters(img, class_vector, k):
  """
  Update the centroid location by taking the average of the points in each cluster group
  :param class_vector: consists of pixels and the clusters they belong
  :param k: #cluster
  :return: updated cluster centers
  """
  im = img
  img = array(im)
  newClusterCenters = []
  rows, cols = img.shape[:2]
  for clusterNum in range(k):
      points = []
      for row in range(rows):
          for col in range(cols):
              if class_vector[row, col] == clusterNum:
                  points.append(img[row, col])
      newClusterCenters.append(np.round(np.mean(points, axis=0), 2))
  return np.float32(newClusterCenters)


def quantize(img, k, mode):
    '''# K-means implementation

    :param img: str, location of image
    :param k: int, #cluster centers
    :param mode: string, choosing initial centroids manually or randomly
    :return: returns the initial centers and class_vector
    '''
    # image's matrix of pixels - the matrix size: height*width*3
    # im = Image.open(img)
    # img_array = array(im)

    # pick k initial color centers manually by clicking on the image
    if mode == 'manual_centers':
        manual_centers = Manual_Centers(img, k)
        initial_centers = manual_centers
    # choose initial k color centers randomly by using numpy.random.uniform
    if mode == 'random_centers':
        random_centers = Random_Centers(img, k)
        initial_centers = random_centers
        # All steps of K-means implementation
    error = 1
    while (error > 0):
        class_vector = ClusteringPixels(img, initial_centers)
        newClusterCenters = updateClusterCenters(img, class_vector, k)
        error = np.linalg.norm(newClusterCenters - initial_centers, axis=None)
        initial_centers = newClusterCenters
    return initial_centers, class_vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noOfCluster", default="2, 4, 8, 16, 32", type=str,
                        help="Number of cluster to be formed")
    parser.add_argument("--image", default='image01.jpeg',
                        help="Image for color quantization")
    parser.add_argument("--modes", default="manual_centers, random_centers")
    parser.add_argument("--colorSpaceType", default="RGB") # LAB
    args, _ = parser.parse_known_args()
    noOfCluster = args.noOfCluster.split(",")
    modes = args.modes.split(",")
    execution_time = {}

    for mode in modes:
        for K in noOfCluster:
            start_time = 0
            end_time = 0
            start_time = time.time()
            K = int(K)
            print("Running image quantization for K={}".format(K))
            if args.colorSpaceType == "LAB":
                image = cv2.imread(args.image)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            else:
                img = cv2.imread(args.image, 1)
            clusterCenter, classificationVector = quantize(img, K, mode)
            rows, cols = img.shape[:2]
            for row in range(rows):
                for col in range(cols):
                    img[row, col] = clusterCenter[classificationVector[row, col]]
            imageBaseName = os.path.basename(args.image)
            filename, fileExt = os.path.splitext(imageBaseName)
            if args.colorSpaceType == "LAB":
                filename = "{}_LAB_{}_{}.png".format(filename, mode, K)
            else:
                filename = "{}_{}_{}.png".format(filename, mode, K)
            cv2.imwrite(filename, img)
            print("Image size after quantization: {}".format(os.path.getsize(filename)))
            end_time = time.time()
            execution_time['K'] = end_time - start_time

print(execution_time)