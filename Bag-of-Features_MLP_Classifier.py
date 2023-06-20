import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import operator
import random
import torch
import cv2
import csv
import os

from kornia_moons.feature import *
import kornia.feature as kf
import kornia

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def listOfImagePaths(images_path):
    image_list = []
    image_labels = []
    folders = os.listdir(images_path)
    for sub_fold in folders:
        for img in os.listdir(os.path.join(images_path, sub_fold)):
            img_path = os.path.join(images_path, os.path.join(sub_fold, img))
            label = sub_fold
            image_list.append(img_path)
            image_labels.append(label)
    return image_list, np.array(image_labels)


def shuffleImages(images, labels):
    random.seed(13)
    zipFile = list(zip(images, labels))
    random.shuffle(zipFile)
    list1, list2 = zip(*zipFile)
    images, labels = list(list1), list(list2)
    return images, labels


def produceDescriptors(img, device, desc_type, num_descriptors, visualize=False):
    detector = cv2.SIFT_create(num_descriptors)
    affine = kf.LAFAffineShapeEstimator(patch_size=32, affine_shape_detector=None, preserve_orientation=True).to(device)
    if desc_type == 'SIFT':
        descriptor = kf.SIFTDescriptor(32, 8, 4).to(device)
    else:
        descriptor = kf.HyNet(pretrained=True).to(device)

    if visualize:
        plt.imshow(img)
    kpts = detector.detect(img, None)[:num_descriptors]

    with torch.no_grad():
        timg = kornia.image_to_tensor(img, False).float() / 255.
        timg = timg.to(device)
        timg_gray = kornia.color.rgb_to_grayscale(timg).to(device)

        lafs = laf_from_opencv_SIFT_kpts(kpts, device=device) # kornia_moons
        lafs_new = affine.forward(lafs, timg_gray)

        if visualize:
            visualize_LAF(timg, lafs_new, 0)

        patches = kf.extract_patches_from_pyramid(timg_gray, lafs_new, 32).to(device)  # what if we change pathch size
        print(patches.shape)
        B, N, CH, H, W = patches.size()
        if N == 0:
            print('kpts are:', kpts)
            print('lafs are:', lafs)
            print('lafs_new are:', lafs_new)
            plt.imshow(img)
            plt.show()
        descs = descriptor(patches.view(B * N, CH, H, W)).detach().cpu().numpy()
    return kpts, descs


def calculateKMeans(descriptionTensors, num_clusters, device):
    """
    :param descriptionTensors: all generated descriptors from the images
    :param num_clusters: #centers for k-means clustering
    :return: clustered indexes of descriptors and found cluster centers
    """
    cluster_ids_x, cluster_centers = kmeans(X=descriptionTensors, num_clusters=num_clusters,
                                            distance='euclidean', device=device)
    return cluster_ids_x, cluster_centers


def featureQuantization(cluster_ids_x, cluster_centers, descriptors, num_images, k, device):
    extractedFeatures = np.array([np.zeros(k) for i in range(num_images)])
    for i in range(num_images):
        features = descriptors[i]
        features = torch.from_numpy(features.reshape(len(descriptors[i]), 128))
        cluster_idxs = kmeans_predict(features, cluster_centers, 'euclidean', device=device)
        for j in cluster_idxs:
            extractedFeatures[i][j] += 1
    return extractedFeatures


def plotHistogram(image_name, quantized_features, no_clusters):
    bins = np.arange(no_clusters)
    bin_counts = np.array([np.sum(quantized_features[:, h], dtype=np.int32) for h in range(no_clusters)])
    # Normalize bin counts
    sum_ = sum(bin_counts)
    binCounts = []
    for i in bin_counts:
        binCounts.append(i/sum_)

    plt.bar(bins, binCounts, align='center', color= "green", edgecolor="purple")
    plt.xlabel("codewords")
    plt.ylabel("Frequency")
    plt.title("Histogram of Generated Features")
    plt.xticks(bins + 0.4, bins)
    plt.savefig('hist_{}.png'.format(image_name))


def calculateStatistics(test_predictions, image_labels, classes):
    # Mean F1 score
    mean_F1 = f1_score(image_labels, test_predictions, labels=np.unique(test_predictions), average='macro')
    # Per class F1 score
    f1_scores = f1_score(image_labels, test_predictions, average=None, labels=np.unique(test_predictions))
    f1_scores_with_labels = {label: score for label, score in zip(classes, f1_scores)}
    # Mean balanced accuracy score
    meanBalancedAcc = balanced_accuracy_score(image_labels, test_predictions)
    # Mean imbalanced accuracy score
    meanImbalancedAcc = accuracy_score(image_labels, test_predictions)

    return mean_F1, f1_scores, f1_scores_with_labels, meanBalancedAcc, meanImbalancedAcc


def confusionMatrix(image_name, test_predictions, image_labels, classes, matrix_size):
    cm = confusion_matrix(image_labels, test_predictions, labels=classes)
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(matrix_size, matrix_size))
    cmp.plot(ax=ax)

    cmp.figure_.savefig('ConfusionMatrix_{}.png'.format(image_name))

    # Per-class accuracy score
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracyScore = cm.diagonal()
    return accuracyScore


def determineResults(predicted, actual_labels, images_list, image_name, f1scores_with_labels):
    # draw 20*20 thumbnail
    thumbnail_images = []
    for idx, c in enumerate(predicted):
        if c != actual_labels[idx]:
            thumbnail_images.append(images_list[idx])
    random.shuffle(thumbnail_images)
    for j, im_path in enumerate(thumbnail_images[:16]):
        image = Image.open(im_path)
        MAX_SIZE = (20, 20)
        image.thumbnail(MAX_SIZE)
        plt.subplot(4, 4, j+1)
        plt.imshow(image)

    plt.savefig('thumbnails_{}.png'.format(image_name))

    # best 5 classes and the worst 5 classes
    sorted_best = dict(sorted(f1scores_with_labels.items(), key=operator.itemgetter(1), reverse=True))
    sorted_worst = dict(sorted(f1scores_with_labels.items(), key=operator.itemgetter(1)))

    best_classes = list(sorted_best.keys())[:6] #6
    worst_classes = list(sorted_worst.keys())[:6] #6

    return best_classes, worst_classes


def trainSVC(image_name, data_dir, no_clusters, feature_descriptor, num_descriptors,  device):
    image_list, image_labels = listOfImagePaths(data_dir)
    images, labels = shuffleImages(image_list, image_labels)
    imageDescriptions = []
    print(len(images))
    for indx, im in enumerate(images):
        image = cv2.imread(im)
        _, imDescp = produceDescriptors(image, device, feature_descriptor, num_descriptors, visualize=False)
        if indx % 1000 == 0:
            print('{} done!!'.format(indx))
        # print(imDescp.shape)
        imageDescriptions.append(imDescp)
    print('Image descriptors computed!!')

    # Concatenate list of descriptor in imageDescriptions
    print('Stacking started!!')
    descriptors = np.array(imageDescriptions[0])
    for indx, descriptor in enumerate(imageDescriptions[1:]):
        descriptors = np.vstack((descriptors, descriptor))
        if indx % 1000 == 0:
            print('{} done!!'.format(indx))
    descriptors = torch.from_numpy(descriptors)
    print('Train descriptors concatenated!', descriptors.shape)

    # #clusters 50, 100 and 500
    cluster_ids_x, cluster_centers = calculateKMeans(descriptors, no_clusters, device)
    print('Kmeans computed!!')

    quantizedFeatures = featureQuantization(cluster_ids_x, cluster_centers, imageDescriptions, len(image_list),
                                            no_clusters, device)
    print("Image features extracted!!")

    plotHistogram(image_name, quantizedFeatures, no_clusters)
    print('Histogram normalized and plotted!!')

    # Scale the values in quantizedFeatures to save computation
    sc_X = StandardScaler()
    scaled_quantizedFeatures = sc_X.fit_transform(quantizedFeatures)

    # train model
    print('Classification started!!')
    classifier = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                               activation="relu", solver='adam', random_state=1).fit(scaled_quantizedFeatures, labels)
    print('Classifier:', classifier)

    return classifier, cluster_ids_x, cluster_centers, scaled_quantizedFeatures, sc_X


def testSVC(data_dir, feature_descriptor, num_descriptors, cluster_ids_x, cluster_centers, nu_clusters,
            classifier, sc_X, device):
    image_list, test_image_labels = listOfImagePaths(data_dir)
    images, labels = image_list, test_image_labels
    imageDescriptions = []
    print(len(images))
    for indx, im in enumerate(images):
        image = cv2.imread(im)
        _, imDescp = produceDescriptors(image, device, feature_descriptor, num_descriptors, visualize=False)
        if indx % 1000 == 0:
            print('{} done!!'.format(indx))
        # print(imDescp.shape)
        imageDescriptions.append(imDescp)
    print('Test image descriptors computed!!')

    testQuantizedFeatures = featureQuantization(cluster_ids_x, cluster_centers, imageDescriptions, len(images),
                                                nu_clusters, device)
    print("Test image features extracted!!")

    # transformed quantized features
    features = sc_X.transform(testQuantizedFeatures)
    # predict test classes
    predicted_labels = classifier.predict(features)
    classes = classifier.classes_
    print('classes are:', classes)
    return predicted_labels, labels, image_list, classes


def run(dataset_name, data_dir, test_dir, feature_descriptor, num_descriptors, no_clusters, matrix_size):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image_name = '{}_{}_{}_{}'.format(dataset_name, feature_descriptor, num_descriptors, no_clusters)

    multiclass_classifier, cluster_ids_x, cluster_centers, quantizedFeatures, sc_x = \
        trainSVC(image_name, data_dir, no_clusters, feature_descriptor, num_descriptors, device)
    print('Training COMPLETED!!')

    testPredictions, test_image_labels, image_list, classes = \
        testSVC(test_dir, feature_descriptor, num_descriptors, cluster_ids_x, cluster_centers, no_clusters,
                multiclass_classifier, sc_x, device)
    print('Test COMPLETED')
    print('type is:', len(testPredictions))

    print('Statistical Calculations started!!')
    classAccuracies = confusionMatrix(image_name, testPredictions, test_image_labels, classes, matrix_size)
    print('classAccuracies', classAccuracies)
    mean_F1, f1_scores, f1_scores_with_labels, meanBalancedAcc, meanImbalancedAcc = \
        calculateStatistics(testPredictions, test_image_labels, classes)
    print('mean_F1', mean_F1)
    print('f1_scores', f1_scores)
    print('f1_scores_with_labels', f1_scores_with_labels)
    print('meanBalancedAcc', meanBalancedAcc)
    print('meanImbalancedAcc', meanImbalancedAcc)

    # plot 20*20 thumbnail and best and worst classes
    best_classes, worst_classes = determineResults(list(testPredictions), list(test_image_labels), image_list, image_name,
                                                   f1_scores_with_labels)

    parameters = ['classAccuracies', 'mean_F1', 'meanBalancedAcc', 'meanImbalancedAcc', 'f1_scores_with_labels',
                  'f1_scores', 'best_classes', 'worst_classes']
    results = [
        {'classAccuracies': classAccuracies, 'mean_F1': mean_F1, 'meanBalancedAcc': meanBalancedAcc,
         'meanImbalancedAcc': meanImbalancedAcc,
         'best_classes': best_classes, 'worst_classes': worst_classes}]

    with open('outputs_{}.csv'.format(image_name), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=parameters)
        writer.writeheader()
        writer.writerows(results)

    plt.plot(multiclass_classifier.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    return print('FINISHED!!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--dataDir", type=str)
    parser.add_argument("--testDir", type=str)
    parser.add_argument("--featureDescriptor", type=str)  # Hynet, SIFT
    parser.add_argument("--nuOfDescriptors", type=int)
    parser.add_argument("--noOfClusters", type=int)
    parser.add_argument("--confusionMatrixSize", type=int)
    args = vars(parser.parse_args())

    run(args['name'], args['dataDir'], args['testDir'], args['featureDescriptor'], args['nuOfDescriptors'],
        args['noOfClusters'], args['confusionMatrixSize'])

