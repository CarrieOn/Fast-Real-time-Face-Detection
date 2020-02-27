import argparse
import os
import glob
import scipy.misc
import cv2
import copy
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from utils import pickle_load, image_scipy_to_cv, display_image, compute_intersection_area
from utils import RectangleRegion, DetectionResult, DetectionResults

class CandidateRegion(object):
    def __init__(self, image_np, x, y, window_size, scale):
        self.image_np = image_np
        self.x = x
        self.y = y
        self.window_size = window_size
        self.scale = scale

class Detector(object):
    def __init__(self, classifier_filename, detect_window_size=24, detect_window_step_size=24, n_process=1):
        self.classifier = pickle_load(classifier_filename)
        self.window_size = detect_window_size
        self.step_size = detect_window_step_size
        self.n_process = n_process
        self.pool = None if self.n_process <=1 else Pool(self.n_process)

    def detect(self, image_filename, debug=False):
        image_np = scipy.misc.imread(image_filename, mode='F')
        if self.n_process == 1:
            return self.scan_image_with_scale(image_np, debug=debug)
        else:
            return self.scan_image_with_scale_mp(image_np, debug=debug)

    @staticmethod
    def detect_candidate(candidate, classifier):
        return classifier.classify(candidate.image_np[candidate.y:candidate.y+candidate.window_size, candidate.x:candidate.x+candidate.window_size])

    def scan_image_with_scale_mp(self, image_np, min_size=0.0, max_size=1.0, scale_step=0.6, debug=False):
        height, width = image_np.shape

        if min_size * min(width, height) < self.window_size:
            min_size = float(self.window_size) / min(width, height)
        assert max_size > min_size, "max_size: {}, min_size: {}".format(max_size, min_size)
        assert scale_step > 0.0 and scale_step < 1.0

        candidates = []
        si = max_size
        while True:
            scale_image_np = scipy.misc.imresize(image_np, size=(int(si * height), int(si * width)), mode='F')
            h, w = scale_image_np.shape
            for y in range(0, h - self.window_size, self.step_size):
                for x in range(0, w - self.window_size, self.step_size):
                    candidates.append(CandidateRegion(scale_image_np, x, y, self.window_size, si))
            if si <= min_size: break
            si *= scale_step
            if si < min_size:
                si = min_size

        if debug:
            print("Collected %d candidate regions" % len(candidates))
            predictions = self.pool.map(partial(self.detect_candidate, classifier=self.classifier), tqdm(candidates, total=len(candidates), desc="detect"))
        else:
            predictions = self.pool.map(partial(self.detect_candidate, classifier=self.classifier), candidates)
        
        detected_regions = []
        #for candidate, pred in tqdm(zip(candidates, predictions), desc="get result"):
        for candidate, pred in zip(candidates, predictions):
            if pred >= 0.5:
                detected_regions.append(RectangleRegion(candidate.x / candidate.scale, candidate.y / candidate.scale, candidate.window_size / candidate.scale, candidate.window_size / candidate.scale))

        merged_regions = self.merge_regions(detected_regions)
        if debug:
            self.display_regions(image_np, detected_regions, merged_regions)

        return merged_regions

    def scan_image(self, image_np, current_scale):
        h, w = image_np.shape
        detected_regions = []
        for y in range(0, h - self.window_size, self.step_size):
            for x in range(0, w - self.window_size, self.step_size):
                pred = self.classifier.classify(image_np[y:y+self.window_size, x:x+self.window_size])
                if pred >= 0.5:
                    detected_regions.append(RectangleRegion(x / current_scale, y / current_scale, self.window_size / current_scale, self.window_size / current_scale))
        return detected_regions
    
    def scan_image_with_scale(self, image_np, min_size=0.0, max_size=1.0, scale_step=0.6, debug=False):
        height, width = image_np.shape

        if min_size * min(width, height) < self.window_size:
            min_size = float(self.window_size) / min(width, height)
        assert max_size > min_size, "max_size: {}, min_size: {}".format(max_size, min_size)
        assert scale_step > 0.0 and scale_step < 1.0

        detected_regions = []
        si = max_size
        while True:
            scale_image_np = scipy.misc.imresize(image_np, size=(int(si * height), int(si * width)), mode='F')
            detected_regions.extend(self.scan_image(scale_image_np, current_scale=si))
            if si <= min_size: break
            si *= scale_step
            if si < min_size:
                si = min_size
        
        merged_regions = self.merge_regions(detected_regions)
        if debug:
            self.display_regions(image_np, detected_regions, merged_regions)

        return merged_regions

    def display_regions(self, image_np, detected_regions, merged_regions):
        img_cv = image_scipy_to_cv(image_np)
        for region in detected_regions:
            img_cv = cv2.rectangle(img_cv, (int(region.x), int(region.y)), (int(region.x+region.width), int(region.y+region.height)), (0, 255, 0), 2)
        for region in merged_regions:
            img_cv = cv2.rectangle(img_cv, (int(region.x), int(region.y)), (int(region.x+region.width), int(region.y+region.height)), (0, 0, 255), 2)
        display_image(img_cv)

    def merge_regions(self, regions):
        if len(regions) <= 1:
            return regions

        region_to_set_idx = {}
        set_idx = 0
        set_list = []

        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                a = regions[i]
                b = regions[j]
                if compute_intersection_area(a, b) > 0:
                    if i not in region_to_set_idx and j not in region_to_set_idx:
                        # create new set
                        region_to_set_idx[i] = set_idx
                        region_to_set_idx[j] = set_idx
                        new_set = set()
                        new_set.add(i)
                        new_set.add(j)
                        set_list.append(new_set)
                        set_idx += 1
                    elif i in region_to_set_idx and j not in region_to_set_idx:
                        # set j to i's set
                        region_to_set_idx[j] = region_to_set_idx[i]
                        set_list[region_to_set_idx[i]].add(j)
                    elif i not in region_to_set_idx and j in region_to_set_idx:
                        # set i to j's set
                        region_to_set_idx[i] = region_to_set_idx[j]
                        set_list[region_to_set_idx[j]].add(i)
                    elif region_to_set_idx[i] == region_to_set_idx[j]:
                        # already in a set, nothing needs to be done
                        pass
                    else:
                        # need to merge i and j's sets
                        src_set_idx = max(region_to_set_idx[i], region_to_set_idx[j])
                        tgt_set_idx = min(region_to_set_idx[i], region_to_set_idx[j])
                        for k in range(len(regions)):
                            if k in region_to_set_idx and region_to_set_idx[k] == src_set_idx:
                                region_to_set_idx[k] = tgt_set_idx
                                set_list[src_set_idx].remove(k)
                                set_list[tgt_set_idx].add(k)
            if i not in region_to_set_idx:
                region_to_set_idx[i] = set_idx
                new_set = set()
                new_set.add(i)
                set_list.append(new_set)
                set_idx += 1

        result = []
        for current_set in set_list:
            current_region_indices = list(current_set)
            if len(current_region_indices) == 0:
                continue
            elif len(current_region_indices) == 1:
                result.append(regions[current_region_indices[0]])
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
            for region_idx in current_region_indices:
                region = regions[region_idx]
                x1 += region.x
                y1 += region.y
                x2 += (region.x + region.width)
                y2 += (region.y + region.height)
            x = x1 / len(current_region_indices)
            y = y1 / len(current_region_indices)
            width = x2 / len(current_region_indices) - x
            height = y2 / len(current_region_indices) - y
            result.append(RectangleRegion(x, y, width, height))

        return result

def debug_detector():
    detector = Detector(os.path.join('models', 'strong_classifier_276.pkl'), n_process=6)
    for image_file in glob.glob(os.path.join("../test_images_1000", "*.jpg")):
        #detected_faces = detector.detect(image_file)
        detector.detect(image_file, debug=True)
    
def get_detection_results(path, debug=False):
    detector = Detector(os.path.join('models', 'strong_classifier_276.pkl'), n_process=10)

    results = DetectionResults()
    for image_file in tqdm(glob.glob(os.path.join(path, "*.jpg"))):
        image_cv = cv2.imread(image_file)
        detected_faces = detector.detect(image_file, debug=False)
        result = DetectionResult(image_file)
        for region in detected_faces:
            result.add_bbox(region)
            if debug:
                image_cv = cv2.rectangle(image_cv, (int(region.x), int(region.y)), (int(region.x+region.width), int(region.y+region.height)), (0, 255, 0), 2)
        results.add(result)
        if debug:
            display_image(image_cv)

    return results

if __name__ == "__main__":
    debug_detector()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    
    detection_results = get_detection_results(args.data_dir)
    detection_results.save('result_xxx.json')