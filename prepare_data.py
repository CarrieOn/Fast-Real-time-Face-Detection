
import os
import sys
import glob
import math
import numpy as np
import scipy.misc
import cv2
import pandas as pd
import argparse
import copy
from tqdm import tqdm
from PIL import Image
import random
import sklearn.utils
from utils import pickle_save, pickle_load, image_scipy_to_cv, display_image, compute_intersection_area, compute_iou
from utils import RectangleRegion

class FddbFaceEllipse(object):
    def __init__(self, ellipse_line):
        self.major_radius, self.minor_radius, self.angle, self.center_x, self.center_y, self.detection_score = map(float, ellipse_line.split())

class FddbImageResult(object):
    def __init__(self, image_filename, is_missing=False, image_np=None, image_cv=None):
        self.image_filename = image_filename
        self.is_missing = is_missing
        self.image_np = image_np
        self.image_cv = image_cv
        self.faces = []

    def add_face(self, face):
        self.faces.append(face)

class FddbImageIterator(object):
    def __init__(self, list_path, image_path, skip_missing=False):
        self.list_path = list_path
        self.image_path = image_path
        self.skip_missing = skip_missing

    def __iter__(self):
        for list_file in os.listdir(self.list_path):
            if list_file.endswith("ellipseList.txt"):
                all_lines = open(os.path.join(self.list_path, list_file)).readlines()
                all_lines = [l.rstrip("\n") for l in all_lines]
                idx = 0

                while idx < len(all_lines):
                    img_filename = all_lines[idx]
                    face_count = int(all_lines[idx + 1])
                    idx += 2
                    if not img_filename:
                        break

                    try:
                        image_fullpath = os.path.join(self.image_path, img_filename + ".jpg")
                        img_np = scipy.misc.imread(image_fullpath, mode='F')
                        img_cv = cv2.imread(image_fullpath)
                    except FileNotFoundError:
                        print("Image %s not found!" % image_fullpath)
                        if not self.skip_missing:
                            yield FddbImageResult(image_fullpath, is_missing=True)
                        continue

                    image_result = FddbImageResult(image_fullpath, image_np=img_np, image_cv=img_cv)
                    for _ in range(face_count):
                        image_result.add_face(FddbFaceEllipse(all_lines[idx]))
                        idx += 1
                    yield image_result

def filter_coordinate(c, m):
    if c < 0:
    	return 0
    elif c > m:
    	return m
    else:
    	return c

def get_rectangle_coordiates(width, height, major_axis_radius, minor_axis_radius, angle, center_x, center_y, square=True, square_mode="average"):
    tan_t = -(minor_axis_radius/major_axis_radius) * math.tan(angle)
    t = math.atan(tan_t)
    x1 = center_x + (major_axis_radius * math.cos(t) * math.cos(angle) - minor_axis_radius * math.sin(t) * math.sin(angle))
    x2 = center_x + (major_axis_radius * math.cos(t + math.pi) * math.cos(angle) - minor_axis_radius * math.sin(t + math.pi) * math.sin(angle))
    x_max = filter_coordinate(max(x1, x2), width)
    x_min = filter_coordinate(min(x1, x2), width)

    tan_t = (minor_axis_radius/major_axis_radius) * (1/(math.tan(angle) + 1e-8))
    t = math.atan(tan_t)
    y1 = center_y + (minor_axis_radius * math.sin(t) * math.cos(angle) + major_axis_radius * math.cos(t) * math.sin(angle))
    y2 = center_y + (minor_axis_radius * math.sin(t + math.pi) * math.cos(angle) + major_axis_radius * math.cos(t + math.pi) * math.sin(angle))
    y_max = filter_coordinate(max(y1, y2), height)
    y_min = filter_coordinate(min(y1, y2), height)

    if square:
        x_c = (x_max + x_min) / 2
        y_c = (y_max + y_min) / 2
        w = x_max - x_min
        h = y_max - y_min
        if square_mode == "average":
            w = (w + h) / 2
        elif square_mode == "max":
            w = max(w, h)
        elif square == "min":
            w = min(w, h)
        
        w = 2 * min(w/2, x_c, width - x_c, y_c, height - y_c)
        x_min = x_c - w/2
        x_max = x_c + w/2
        y_min = y_c - w/2
        y_max = y_c + w/2

    return tuple(map(int, [x_min, y_min, x_max, y_max]))

def generate_positive(list_path, image_path, out_path, show=False):
    """
    Generate positive examples using lists from list_path, using image from img_path,
    and save to out_path
    """
    missing_file_count, face_idx, too_small_count = 0, 0, 0
    for img_result in FddbImageIterator(list_path, image_path):
        if img_result.is_missing:
            missing_file_count += 1
            continue
        
        height, width = img_result.image_np.shape
        for face in img_result.faces:
            x_min, y_min, x_max, y_max = get_rectangle_coordiates(width, height, face.major_radius, face.minor_radius, face.angle, face.center_x, face.center_y, square=True)

            if show:
                img_cv = cv2.rectangle(img_result.image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                img_cv = cv2.ellipse(img_cv, (int(face.center_x), int(face.center_y)), (int(face.major_radius), int(face.minor_radius)), int(face.angle / math.pi * 180), 0, 360, (0,0,255), 2)
                print(face.major_radius, face.minor_radius, face.angle, face.center_x, face.center_y)
                print(width, height)
                print(x_min, y_min, x_max, y_max)

            if (x_max - x_min) < 24:
                too_small_count += 1
                continue

            try:
                base, ext = os.path.splitext(os.path.basename(img_result.image_filename))
                out_image_name = base + "-" + str(face_idx) + ".jpg"
                face_idx += 1
                scipy.misc.imsave(os.path.join(out_path, out_image_name), img_result.image_np[y_min:y_max, x_min:x_max])
                print('Face image %s generated.' % out_image_name)
            except Exception as ex:
                print(ex)

        if show:
            display_image(img_cv, img_result.image_filename)
    print('Face generation done with %d faces generated. %d files missing. %d too small images' % (face_idx, missing_file_count, too_small_count))

def overlap_with_face(region, face_regions, overlap_threshold=0.01):
    region_area = region.width * region.height
    if region_area <= 0:
        rerun False
    for face_region in face_regions:
        intersectin_area = compute_intersection_area(region, face_region)
        overlap_ratio = float(intersectin_area) / region_area
        if overlap_ratio >= overlap_threshold:
            return True
    return False

def scan_image(image_np, window_size, window_step_size, face_regions=[], debug=False):
    h, w = image_np.shape
    data = []
    for y in range(0,h-window_size,window_step_size):
        for x in range(0,w-window_size,window_step_size):
            overlap = overlap_with_face(RectangleRegion(x, y, window_size, window_size), face_regions)
            if debug:
                img_cv = image_scipy_to_cv(image_np)
                color = (0, 0, 255) if overlap else (0, 255, 0)
                img_cv = cv2.rectangle(img_cv, (x, y), (x+window_size, y+window_size), color, 2)
                display_image(img_cv)
            if not overlap:
                data.append(image_np[y:y+window_size, x:x+window_size])
    return data

def debug_scan_image():
    img_np = scipy.misc.imread('test/img_197.jpg', mode='F')
    scan_image(img_np, 24, 48, debug=True)

def scan_image_with_scale(image_np, window_size=24, window_step_size=48, min_size=0.0, max_size=1.0, scale_step=0.5, face_regions=[], debug=False):
    height, width = image_np.shape
    data = []

    if min_size * min(width, height) < window_size:
        min_size = float(window_size) / min(width, height)
    assert max_size > min_size, "max_size: {}, min_size: {}".format(max_size, min_size)
    assert scale_step > 0.0 and scale_step < 1.0

    si = max_size
    while True:
        scale_image_np = scipy.misc.imresize(image_np, size=(int(si * height), int(si * width)), mode='F')
        scale_face_regions = [copy.deepcopy(r).scale(si) for r in face_regions]
        #import pdb; pdb.set_trace()
        scale_data = scan_image(scale_image_np, window_size, window_step_size, scale_face_regions, debug=debug)
        data.extend(scale_data)
        if si <= min_size: break
        si *= scale_step
        if si < min_size:
            si = min_size

    return data

def debug_scan_image_with_scale():
    img_np = scipy.misc.imread('test/img_197.jpg', mode='F')
    scan_image_with_scale(img_np, 100, 100, scale_step=0.75, debug=True)

def generate_negative_from_face_dataset(list_path, image_path, out_path, show=False, random_file_ratio=0.3, random_crop_ratio=0.9, max_count=3000):
    neg_idx = 0
    file_count = 0
    for img_result in FddbImageIterator(list_path, image_path, skip_missing=True):
        if random.random() > random_file_ratio:
            continue
        
        print("Reading file %s" % img_result.image_filename)
        height, width = img_result.image_np.shape
        face_regions = []
        for face in img_result.faces:
            x_min, y_min, x_max, y_max = get_rectangle_coordiates(width, height, face.major_radius, face.minor_radius, face.angle, face.center_x, face.center_y, square=True, square_mode="max")
            region = RectangleRegion(x_min, y_min, x_max - x_min, y_max - y_min)
            face_regions.append(region)

        data = scan_image_with_scale(img_result.image_np, window_size=50, window_step_size=60, min_size=0.0, max_size=1.0, scale_step=0.5, face_regions=face_regions, debug=show)

        if show:
            image_cv = image_scipy_to_cv(img_result.image_np)
            display_image(image_cv)

        counter = 0
        for neg_np in data:
            if random.random() > random_crop_ratio:
                continue
            counter += 1
            base, ext = os.path.splitext(os.path.basename(img_result.image_filename))
            out_image_name = os.path.join(out_path, base + "-" + str(neg_idx) + ".jpg")
            neg_idx += 1
            #neg_np_scale = scipy.misc.imresize(neg_np, size=(24, 24), mode='F')
            scipy.misc.imsave(out_image_name, neg_np)
        print("Generated %d/%d images" % (counter, len(data)))
        if counter > 0:
            file_count += 1
        if neg_idx > max_count:
            break
    print("Generated %d negative images altogether from %d files" % (neg_idx, file_count))

def generate_negative_from_non_face_dataset(image_path, out_path, show=True, random_file_ratio=0.1, random_crop_ratio=0.5, max_count=3000):
    neg_idx = 0
    file_count = 0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for subdir, dirs, files in os.walk(image_path):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            if random.random() > random_file_ratio:
                    continue

            filepath = subdir + os.sep + file

            img = cv2.imread(filepath)
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
            if len(detected_faces) > 0:
                continue

            image_np = scipy.misc.imread(filepath, mode='F')
            data = scan_image_with_scale(image_np, window_size=50, window_step_size=100, min_size=0.0, max_size=1.0, scale_step=0.5, face_regions=[], debug=show)

            counter = 0
            for neg_np in data:
                if random.random() > random_crop_ratio:
                    continue
                counter += 1
                base, ext = os.path.splitext(os.path.basename(filepath))
                out_image_name = os.path.join(out_path, base + "-" + str(neg_idx) + ".jpg")
                neg_idx += 1
                scipy.misc.imsave(out_image_name, neg_np)
            print("Generated %d/%d images" % (counter, len(data)))
            if counter > 0:
                file_count += 1
            if neg_idx > max_count:
                print("Generated %d negative images altogether from %d files" % (neg_idx, file_count))
                return
    print("Generated %d negative images altogether from %d files" % (neg_idx, file_count))

def test_image_bbox(list_path, image_path, show=False):
    # Verified that angle in the ellipseList.txt file is in radians
    # max 1.570796, min -1.570796, mean 0.158146
    face_counts = 0
    all_angles = []
    for img_result in tqdm(FddbImageIterator(list_path, image_path, skip_missing=True)):
        img_cv = img_result.image_cv
        face_counts += len(img_result.faces)
        for face in img_result.faces:
            all_angles.append(face.angle)
            angle_degree = face.angle / math.pi * 180
            img_cv = cv2.ellipse(img_cv, (int(face.center_x), int(face.center_y)), (int(face.major_radius), int(face.minor_radius)), int(angle_degree), 0, 360, (0,0,255), 2)
        if show:
            display_image(img_cv, img_result.image_filename)
    ps = pd.Series(all_angles)
    print(ps.describe())
    print("Total face count: {}".format(face_counts)) # 5171

def save_image_data(pos_folders, neg_folders, output_folder, target_window_size=24, test_size=400, valid_size=100):
    pos_data = []
    neg_data = []
    for folder in pos_folders:
        for file in os.listdir(folder):
            if not file.endswith(".jpg"):
                continue
            img_np = scipy.misc.imread(os.path.join(folder, file), mode='F')
            img_resize = scipy.misc.imresize(img_np, size=(target_window_size, target_window_size), mode='F')
            pos_data.append(img_resize)
    for folder in neg_folders:
        for file in os.listdir(folder):
            if not file.endswith(".jpg"):
                continue
            img_np = scipy.misc.imread(os.path.join(folder, file), mode='F')
            img_resize = scipy.misc.imresize(img_np, size=(target_window_size, target_window_size), mode='F')
            neg_data.append(img_resize)
    
    random.seed(33)
    random.shuffle(pos_data)
    random.shuffle(neg_data)

    test_pos = pos_data[:test_size]
    valid_pos = pos_data[test_size:test_size+valid_size]
    train_pos = pos_data[test_size+valid_size:]
    test_neg = neg_data[:test_size]
    valid_neg = neg_data[test_size:test_size+valid_size]
    train_neg = neg_data[test_size+valid_size:]
    X_train = np.array(train_pos + train_neg)
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
    X_valid = np.array(valid_pos + valid_neg)
    y_valid = np.array([1] * len(valid_pos) + [0] * len(valid_neg))
    X_test = np.array(test_pos + test_neg)
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=33)
    X_valid, y_valid = sklearn.utils.shuffle(X_valid, y_valid, random_state=33)
    X_test, y_test = sklearn.utils.shuffle(X_test, y_test, random_state=33)
    print("Train all/pos/neg: {}/{}/{}".format(X_train.shape[0], len(train_pos), len(train_neg)))
    print("Valid all/pos/neg: {}/{}/{}".format(X_valid.shape[0], len(valid_pos), len(valid_neg)))
    print("Test all/pos/neg: {}/{}/{}".format(X_test.shape[0], len(test_pos), len(test_neg)))
    """
    Train all/pos/neg: 9732/4214/5518
    Valid all/pos/neg: 200/100/100
    Test all/pos/neg: 800/400/400
    """

    pickle_save((X_train, y_train), os.path.join(output_folder, "train.pkl"))
    pickle_save((X_valid, y_valid), os.path.join(output_folder, "valid.pkl"))
    pickle_save((X_test, y_test), os.path.join(output_folder, "test.pkl"))

def debug_saved_pkl():
    X_train, y_train = pickle_load('train.pkl')
    print(y_train[:10])
    scipy.misc.imsave('tmp.jpg', X_train[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #debug_scan_image()
    #debug_scan_image_with_scale()
    #debug_saved_pkl()
    parser.add_argument("--mode")
    args = parser.parse_args()
    
    if args.mode == "pos":
        generate_positive('../FDDB/FDDB-folds', '../FDDB/originalPics', '../FDDB/processed/test', show=False)
    elif args.mode == "neg":
        generate_negative_from_face_dataset('../FDDB/FDDB-folds', '../FDDB/originalPics', '../FDDB/processed/test', show=False)
    elif args.mode == "neg_nf":
        generate_negative_from_non_face_dataset('../101_ObjectCategories', '../FDDB/processed/test', show=False)
    elif args.mode == "test":
        test_image_bbox('../FDDB/FDDB-folds', '../FDDB/originalPics')
    elif args.mode == "save":
        save_image_data(['../FDDB/processed/faces_filter_manual_remove_vague'], ['../FDDB/processed/non_face', '../FDDB/processed/non_face_101'], '.')
