
import os
import sys
import json
import numpy as np
import pickle
import logging
import cv2
import scipy.misc

logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def image_scipy_to_cv(image_np):
    tmp_file = 'tmp.jpg'
    scipy.misc.imsave(tmp_file, image_np)
    return cv2.imread(tmp_file)

def display_image(image_cv, image_name="unknown"):
    cv2.imshow(image_name, image_cv)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if chr(key & 255) == 'q':
        sys.exit(0)

def compute_intersection_area(a, b):
    x_left = max(a.x, b.x)
    y_top = max(a.y, b.y)
    x_right = min(a.x + a.width, b.x + b.width)
    y_bottom = min(a.y + a.height, b.y + b.height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def compute_iou(a, b):
    x_left = max(a.x, b.x)
    y_top = max(a.y, b.y)
    x_right = min(a.x + a.width, b.x + b.width)
    y_bottom = min(a.y + a.height, b.y + b.height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    a_area = (a.width * a.height)
    b_area = (b.width * b.height)

    iou = intersection_area / float(a_area + b_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def pickle_save(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def pickle_load(file):
    if not os.path.exists(file):
        print("WARNING: file %s does not exist!" % file)
        return None
    with open(file, "rb") as f:
        return pickle.load(f)

def compute_integral_image(image):
    """Get the integral image.

    Integral Image:
    + - - - - -        + -  -  -  -  -  -
    | 1 2 3 4 .        | 0  0  0  0  0  .
    | 5 6 7 8 .   =>   | 0  1  3  6 10  .
    | . . . . .        | 0  6 14 24 36  .
                       | .  .  .  .  .  .
    """
    height, width = image.shape
    ii = np.zeros((height+1, width+1))
    s = np.zeros((height+1, width+1))
    for y in range(1, height+1):
        for x in range(1, width+1):
            s[y][x] = s[y-1][x] + image[y-1][x-1]
            ii[y][x] = ii[y][x-1] + s[y][x]
    return ii

class RectangleRegion(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

    def scale(self, si):
        self.x = int(self.x * si)
        self.y = int(self.y * si)
        self.width = int(self.width * si)
        self.height = int(self.height * si)
        return self

    def area(self):
        return (self.width * self.height)

    def __str__(self):
        return "(x=%d, y=%d, width=%d, height=%d)" % (self.x, self.y, self.width, self.height)
    
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)

    def __eq__(self, other):
        if not isinstance(other, RectangleRegion):
            return False
        return (self.x == other.x and self.y == other.y and self.width == other.width and self.height == other.height)

    def __ne__(self, other):
        return not self.__eq__(other)

class ImageFeature(object):
    def __init__(self, grey_regions, white_regions):
        self.grey_regions = grey_regions
        self.white_regions = white_regions

    def compute(self, image_integral):
        grey_sum = sum([region.compute_feature(image_integral) for region in self.grey_regions])
        white_sum = sum([region.compute_feature(image_integral) for region in self.white_regions])
        return (grey_sum - white_sum)

    def __str__(self):
        return "(grey_regions=%s, white_regions=%s)" % (str(self.grey_regions), str(self.white_regions))
    
    def __repr__(self):
        return "ImageFeature(grey_regions=%s, white_regions=%s)" % (str(self.grey_regions), str(self.white_regions))

    def __eq__(self, other):
        if not isinstance(other, ImageFeature):
            return False
        return (self.grey_regions == other.grey_regions and self.white_regions == other.white_regions)

    def __ne__(self, other):
        return not self.__eq__(other)

class Metrics(object):
    def __init__(self, predictions, labels, confidence_threshold=0.5):
        total_len = len(predictions)
        assert total_len == len(labels)

        n_correct = 0.0
        tp, fp, fn = 0.0, 0.0, 0.0
        for conf, label in zip(predictions, labels):
            pred = 1 if conf >=confidence_threshold else 0
            if pred == label:
                n_correct += 1
                if label == 1:
                    tp += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    fn += 1
        
        self.accuracy = n_correct / total_len
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.detection_rate = self.recall
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.fp_rate = 1 - self.precision

    def __str__(self):
        return "accuracy: {}, precision: {}, recall: {}, f1: {}, dr: {}, fp_rate: {}".format(self.accuracy, self.precision, self.recall, self.f1, self.detection_rate, self.fp_rate)

class DetectionResult(object):
    def __init__(self, iname):
        self.iname = iname
        self.bbox = [] # list[RectangleRegion]

    def add_bbox(self, region):
        self.bbox.append(region)

    def to_dict_list(self):
        result_dict = {}
        result_dict["iname"] = self.iname
        
        bbox_dict_list = []
        for region in self.bbox:
            bbox = [region.x, region.y, region.width, region.height]
            if not isinstance(region.x, int) and not isinstance(region.x, float):
                bbox = [x.item() for x in bbox]
            bbox_dict = {"iname": os.path.basename(self.iname), "bbox": bbox}
            bbox_dict_list.append(bbox_dict)
        return bbox_dict_list

class DetectionResults(object):
    def __init__(self):
        self.result_list = []

    def add(self, result):
        self.result_list.extend(result.to_dict_list())

    def save(self, path):
        with open(path, 'w') as writer:
            json.dump(self.result_list, writer)

def debug_compute_integral_image():
    a = np.random.randint(0, 10, (3,5))
    #a = np.array([[1,2,3,4],[5,6,7,8]])
    b = compute_integral_image(a)
    print(a)
    print(b)

def debug_image_feature():
    a = np.array([[1,2,3,4],[5,6,7,8]])
    b = compute_integral_image(a)
    grey_regions = [RectangleRegion(1,0,2,1)]
    white_regions = [RectangleRegion(1,1,2,1)]
    feature1 = ImageFeature(grey_regions, white_regions)
    feature2 = ImageFeature(grey_regions, white_regions)
    pickle_save(feature1, "test_feature.pkl")
    feature_reload = pickle_load("test_feature.pkl")
    assert feature1 == feature2
    assert feature1 == feature_reload 
    print(a)
    print(feature1)
    print(feature1.compute(b))
    print([feature1, feature2])

if __name__ == "__main__":
    #debug_compute_integral_image()
    debug_image_feature()