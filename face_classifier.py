import os
import numpy as np
import math
from utils import compute_integral_image

class ImageClassifier(object):
    def classify(self, image):
        raise NotImplementedError

class CascadeClassifier(ImageClassifier):
    def __init__(self):
        self.strong_classifiers = []
        self.thresholds = []

    def classify(self, image, is_integral=False):
        for clf, threshold in zip(self.strong_classifiers, self.thresholds):
            if clf.classify(image, is_integral=is_integral) < threshold:
                return 0
        return 1

    def classify_batch(self, image_integrals):
        pred = []
        for image_integral in image_integrals:
            pred.append(self.classify(image_integral, is_integral=True))
        return pred

class AdaBoostClassifier(ImageClassifier):
    def __init__(self):
        self.alphas = []
        self.weak_classifiers = []

    def classify(self, image, is_integral=False):
        total = 0
        image_integral = image if is_integral else compute_integral_image(image)
        for alpha, clf in zip(self.alphas, self.weak_classifiers):
            total += alpha * clf.classify(image_integral)
        return (total / sum(self.alphas))

    def classify_batch(self, image_integrals):
        pred = []
        for image_integral in image_integrals:
            pred.append(self.classify(image_integral, is_integral=True))
        return pred

class WeakClassifier(ImageClassifier):
    def __init__(self, image_feature, threshold, polarity):
        self.image_feature = image_feature
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, image_integral):
        feature = self.image_feature.compute(image_integral)
        return 1 if self.polarity * feature < self.polarity * self.threshold else 0

    def classify_batch(self, image_integrals):
        pred = []
        for image_integral in image_integrals:
            pred.append(self.classify(image_integral))
        return pred

    def __str__(self):
        return "(image_feature={} threshold={}, polarity={})".format(self.image_feature, self.threshold, self.polarity)
    
    def __repr__(self):
        return "WeakClassifier(image_feature={} threshold={}, polarity={})".format(self.image_feature, self.threshold, self.polarity)
