import os
import argparse
import time
import pickle
import math
import copy
import numpy as np
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool
from utils import pickle_load, pickle_save, compute_integral_image, init_logger, logger
from utils import RectangleRegion, ImageFeature, Metrics
import gc
from face_classifier import WeakClassifier, AdaBoostClassifier, CascadeClassifier

class Dataset(object):
    def __init__(self, X=None, y=None, image_features=None, X_integral=None, X_features=None, X_features_sorted=None, X_features_sorted_indices=None):
        self.X = X
        self.y = y
        self.image_features = image_features
        self.X_integral = X_integral
        self.X_features = X_features
        self.X_features_sorted = X_features_sorted
        self.X_features_sorted_indices = X_features_sorted_indices

    @classmethod
    def load(cls, path, name):
        X = cls.safe_np_load(os.path.join(path, "dataset_X_{}.npy".format(name)))
        y = cls.safe_np_load(os.path.join(path, "dataset_y_{}.npy".format(name)))
        image_features = pickle_load(os.path.join(path, "dataset_image_features_{}.pkl".format(name)))
        X_integral = cls.safe_np_load(os.path.join(path, "dataset_X_integral_{}.npy".format(name)))
        X_features = cls.safe_np_load(os.path.join(path, "dataset_X_features_{}.npy".format(name)))
        X_features_sorted = cls.safe_np_load(os.path.join(path, "dataset_X_features_sorted_{}.npy".format(name)))
        X_features_sorted_indices = cls.safe_np_load(os.path.join(path, "dataset_X_features_sorted_indices_{}.npy".format(name)))
        return cls(X, y, image_features, X_integral, X_features, X_features_sorted, X_features_sorted_indices)

    @staticmethod
    def safe_np_load(path):
        if not os.path.exists(path):
            return None
        return np.load(path)

    def save(self, path, name):
        if self.X is not None:
            np.save(os.path.join(path, "dataset_X_{}.npy".format(name)), self.X)
        if self.y is not None:
            np.save(os.path.join(path, "dataset_y_{}.npy".format(name)), self.y)
        if self.image_features is not None:
            pickle_save(self.image_features, os.path.join(path, "dataset_image_features_{}.pkl".format(name)))
        if self.X_integral is not None:
            np.save(os.path.join(path, "dataset_X_integral_{}.npy".format(name)), self.X_integral)
        if self.X_features is not None:
            np.save(os.path.join(path, "dataset_X_features_{}.npy".format(name)), self.X_features)
        if self.X_features_sorted is not None:
            np.save(os.path.join(path, "dataset_X_features_sorted_{}.npy".format(name)), self.X_features_sorted)
        if self.X_features_sorted_indices is not None:
            np.save(os.path.join(path, "dataset_X_features_sorted_indices_{}.npy".format(name)), self.X_features_sorted_indices)

class FeatureResult(object):
    def __init__(self, error, threshold, polarity):
        self.error = error
        self.threshold = threshold
        self.polarity = polarity

class CascadeClassifierTrainer(object):
    def __init__(self, n_process=1):
        self.n_process = n_process
        self.pool = None if n_process == 1 else Pool(n_process)

    def train(self, train_p, train_n, valid_p, valid_n, fp_target, fp_rates, detection_rates, model_save_path, threshold_step=0.01):
        """
        P: positive examples: np.array(n_pos, 24, 24)
        N: negative examples: np.aaray(n_neg, 24, 24)
        fp_target: target false positive rate
        fp_rates: the maximum acceptable false positive rate per layer
        detection_rates: the mimimum acceptable detection rate per layer
        """
        cascade_classifier = CascadeClassifier()
        boost_classifier_model_prefix = "boost_classifier"

        # prepare data
        P = train_p
        N = train_n
        X_valid = np.concatenate((valid_p, valid_n))
        y_valid = np.concatenate((np.ones(len(valid_p)), np.zeros(len(valid_n))))
        X_valid_integral = np.zeros((X_valid.shape[0], X_valid.shape[1]+1, X_valid.shape[2]+1))
        for i in range(len(X_valid)):
            X_valid_integral[i] = compute_integral_image(X_valid[i])

        prev_fp_rate = 1.0
        prev_detection_rate = 1.0
        layer = 0
        fp_rate = prev_fp_rate
        while fp_rate > fp_target:
            logger.log("Start training cascade classifier layer %d" % layer)
            n_weak = 0

            # load self.boost_trainer with right data
            X_train = np.concatenate((P, N))
            y_train = np.concatenate((np.ones(len(P)), np.zeros(len(N))))
            boost_trainer = Trainer(mp_pool=self.pool)
            boost_trainer.prepare_data(X_train, y_train)

            model_prefix = "{}_layer_{}".format(boost_classifier_model_prefix, layer)
            fp_rate = prev_fp_rate
            while fp_rate > (fp_rates[layer] * prev_fp_rate):
                n_weak += 1

                # train from previous model
                prev_model_name = "{}_{}".format(model_prefix, n_weak-1) if n_weak > 1 else None
                boosted_classifier = boost_trainer.train_boosted(n_weak, model_save_path=model_save_path, model_prefix=model_prefix, train_from_model=prev_model_name)

                # evaluate current cascaded classifier on validation set
                candidate_cascade_classifier = copy.deepcopy(cascade_classifier)
                threshold = 0.5
                candidate_cascade_classifier.strong_classifiers.append(boosted_classifier)
                candidate_cascade_classifier.thresholds.append(threshold)

                predictions = candidate_cascade_classifier.classify_batch(X_valid_integral)
                metrics = Metrics(predictions, y_valid)
                #logger.info(metrics)
                fp_rate = metrics.fp_rate
                detection_rate = metrics.detection_rate

                # decrease threshold for the i-th classifier until detection rate reached
                while detection_rate < (detection_rate[layer] * prev_detection_rate):
                    candidate_cascade_classifier.thresholds[-1] -= threshold_step
                    predictions = candidate_cascade_classifier.classify_batch(X_valid_integral)
                    metrics = Metrics(predictions, y_valid)
                    fp_rate = metrics.fp_rate
                    detection_rate = metrics.detection_rate
            
            cascade_classifier.strong_classifiers.append(boosted_classifier)
            pickle_save(cascade_classifier, os.path.join(model_save_path, "cascade_classifier_%d.pkl" % layer))

            # update the negative set
            new_N = []
            for i in len(N):
                if cascade_classifier.classify(N[i]) == 1:
                    new_N.append(N[i])
            N = np.concatenate(new_N)

            layer += 1
        # end while fp_rate > fp_target

class Trainer(object):
    def __init__(self, mp_pool=None):
        self.pool = mp_pool

    def train_boosted(self, n_weak, model_save_path="models", model_prefix="saved_model", train_from_model=None):
        if not model_save_path and not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        n_examples = self.train_ds.X.shape[0]
        assert n_examples == self.train_ds.y.shape[0]

        if not train_from_model and os.path.exists(train_from_model):
            logger.info("Loading classifier from {}...".format(train_from_model))
            model_dict = pickle_load(train_from_model)
            strong_classifier = model_dict["classifier"]
            weights = model_dict["weights"]
            idx = model_dict["idx"]
            logger.info("Loaded strong classifier with {} weak classifiers, idx: {}".format(len(strong_classifier.weak_classifiers), idx))
        else:
            strong_classifier = AdaBoostClassifier()
            idx = 0

            # initializing example weights
            weights = np.zeros(n_examples)
            n_pos = np.sum(self.train_ds.y == 1)
            n_neg = np.sum(self.train_ds.y == 0)
            for i in range(n_examples):
                if self.train_ds.y[i] == 1:
                    weights[i] = 1.0 / (2 * n_pos)
                else:
                    weights[i] = 1.0 / (2 * n_neg)

        for i in range(idx, n_weak):
            # 1. normalize the weights
            weights_sum = np.sum(weights)
            weights = weights / weights_sum

            # 2 & 3. select best weak classifier with respect to the weighted error
            logger.info("\nStart training %d-th weak classifier..." % i)
            start = time.time()
            #clf, error = self.train_weak(self.train_ds.X, self.train_ds.y, self.train_ds.X_features, weights)
            clf, error = self.train_weak(self.train_ds.X, self.train_ds.y, self.train_ds.X_features_sorted, self.train_ds.X_features_sorted_indices, weights)
            logger.info("Finished in {} seconds".format(time.time() - start))

            # evaluate
            logger.info("Weak classifier train metrics:")
            predictions = clf.classify_batch(self.train_ds.X_integral)
            logger.info(Metrics(predictions, self.train_ds.y))

            # 4. update the weights
            beta = error / (1.0 - error)
            for j in range(n_examples):
                e_j = 0 if (predictions[j] == self.train_ds.y[j]) else 1
                weights[j] = weights[j] * (beta ** (1 - e_j))

            alpha = math.log(1.0 / beta)
            strong_classifier.alphas.append(alpha)
            strong_classifier.weak_classifiers.append(clf)

            if model_save_path:
                save_file_name = os.path.join(model_save_path, "%s_%d.pkl" % (model_prefix, i))
                logger.info("Saving strong classifier to {}".format(save_file_name))
                model_dict = {}
                model_dict["classifier"] = strong_classifier
                model_dict["weights"] = weights
                model_dict["idx"] = idx + 1 # next model idx is (idx + 1)
                pickle_save(strong_classifier, model_dict)

            logger.info("Strong classifier train metrics:")
            predictions = strong_classifier.classify_batch(self.train_ds.X_integral)
            logger.info(Metrics(predictions, self.train_ds.y))

            # evaluate on valid
            logger.info("Weak classifier valid metrics:")
            predictions = clf.classify_batch(self.valid_ds.X_integral)
            logger.info(Metrics(predictions, self.valid_ds.y))
            
            logger.info("Strong classifier valid metrics:")
            predictions = strong_classifier.classify_batch(self.valid_ds.X_integral)
            logger.info(Metrics(predictions, self.valid_ds.y))

        return strong_classifier
    
    @staticmethod
    def _get_sorted_feature_tuples(features, weights, y):
        sorted_features = sorted(zip(weights, features, y), key=lambda x: x[1])
        return list(sorted_features)

    @staticmethod
    def _train_weak_one_feature(input, y, weights, total_pos, total_neg):
        X_feature_sorted, X_feature_sorted_indices = input

        pos_weights, neg_weights = 0, 0
        min_error, best_threshold, best_polarity = float('inf'), None, None
        for f, idx in zip(X_feature_sorted, X_feature_sorted_indices):
            w = weights[idx]
            label = y[idx]
            error1 = neg_weights + (total_pos - pos_weights) # treat all (f_value < f) as positive
            error2 = pos_weights + (total_neg - neg_weights) # treat all (f_value < f) as negative
            error = min(error1, error2)
            if error < min_error:
                min_error = error
                best_threshold = f
                best_polarity = 1 if (error1 < error2) else -1
            if label == 1:
                pos_weights += w
            else:
                neg_weights += w
        return FeatureResult(min_error, best_threshold, best_polarity)

    @staticmethod
    def _train_weak_single_process(X_features_sorted, X_features_sorted_indices, image_features, weights, y, total_pos, total_neg, n_features):
        min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for i in tqdm(range(n_features), desc="train weak"):
            #sorted_feature = sorted(zip(weights, X_feature[i], y), key=lambda x: x[1])
            pos_weights, neg_weights = 0, 0
            #for w, f, label in sorted_feature:
            for f, idx in zip(X_features_sorted[i], X_features_sorted_indices[i]):
                w = weights[idx]
                label = y[idx]
                error1 = neg_weights + (total_pos - pos_weights) # treat all (f_value < f) as positive
                error2 = pos_weights + (total_neg - neg_weights) # treat all (f_value < f) as negative
                error = min(error1, error2)
                if error < min_error:
                    min_error = error
                    best_feature = image_features[i]
                    best_threshold = f
                    best_polarity = 1 if (error1 < error2) else -1
                if label == 1:
                    pos_weights += w
                else:
                    neg_weights += w
        return min_error, best_feature, best_threshold, best_polarity

    def train_weak(self, X, y, X_features_sorted, X_features_sorted_indices, weights):
        n_features, n_examples = X_features_sorted.shape
        assert n_features == len(self.train_ds.image_features)
        assert n_examples == y.shape[0]
        assert n_examples == weights.shape[0]

        total_pos, total_neg = 0.0, 0.0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        if self.pool is None:
            min_error, best_feature, best_threshold, best_polarity = self._train_weak_single_process(X_features_sorted, X_features_sorted_indices, self.train_ds.image_features, weights, y, total_pos, total_neg, n_features)
        else:
            feature_results = self.pool.map(partial(self._train_weak_one_feature, y=y, weights=weights, total_pos=total_pos, total_neg=total_neg), tqdm(zip(X_features_sorted, X_features_sorted_indices), total=n_features, desc="train weak"))
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for i, result in tqdm(enumerate(feature_results), total=n_features, desc="merge result"):
                if result.error < min_error:
                    min_error = result.error
                    best_feature = self.train_ds.image_features[i]
                    best_threshold = result.threshold
                    best_polarity = result.polarity
            feature_results = None
            gc.collect()
        logger.info("min_error: {}, best threshold: {}, best polarity: {}".format(min_error, best_threshold, best_polarity))
        logger.info("best feature: {}".format(best_feature))

        return WeakClassifier(best_feature, best_threshold, best_polarity), min_error

    @staticmethod
    def sort_features(X_features, n_process=1):
        def _sort_feature(X_feature):
            indices = np.arange(len(X_feature))
            sorted_tuples = sorted(zip(X_feature, indices), key=lambda x: x[0])
            sorted_features, sorted_indices = map(list, zip(*sorted_tuples))
            return sorted_features, sorted_indices

        n_features, _ = X_features.shape

        X_features_sorted = np.zeros(X_features.shape)
        X_features_sorted_indices = np.zeros(X_features.shape, dtype=np.int32)
        if n_process == 1:
            # do not use multi-process
            for i in tqdm(range(n_features), desc="sort feature"):
                sorted_features, sorted_indices = _sort_feature(X_features[i])
                X_features_sorted[i] = sorted_features
                X_features_sorted_indices[i] = sorted_indices
        else:
            pool = Pool(n_process)
            results = pool.map(_sort_feature, tqdm(X_features, total=n_features, desc="sort feature"))
            for i, (sorted_features, sorted_indices) in enumerate(results):
                X_features_sorted[i] = sorted_features
                X_features_sorted_indices[i] = sorted_indices

        return X_features_sorted, X_features_sorted_indices

    @staticmethod
    def _get_feature(image_feature, X_integral):
        n_examples = X_integral.shape[0]
        return np.array([image_feature.compute(X_integral[j]) for j in range(n_examples)])

    def _prepare_data(self, X, y, name="Unknown"):
        logger.info("\nPreparing data for " + name)
        n_examples, height, width = X.shape
        assert n_examples == y.shape[0]

        logger.info("Buiding features for window size {}x{}...".format(height, width))
        image_features = self.build_features(height, width)
        image_feature_count = len(image_features)
        logger.info("All feature count: {}".format(image_feature_count))

        logger.info("Computing image integrals...")
        X_integral = np.zeros((n_examples, height+1, width+1))
        for i in range(n_examples):
            X_integral[i] = compute_integral_image(X[i])
        logger.info("Finished computing image integrals!")
        
        logger.info("Computing image feature values")
        if self.pool is not None:
            feature_results = self.pool.map(partial(self._get_feature, X_integral=X_integral), tqdm(image_features, total=image_feature_count))
            X_feature = np.array(feature_results)
        else:
            X_feature = np.zeros((image_feature_count, n_examples))
            for i, image_feature in tqdm(enumerate(image_features), total=image_feature_count):
                X_feature[i] = np.array([image_feature.compute(X_integral[j]) for j in range(n_examples)])
        logger.info("Finished computing image feature values!")

        # prepare sorted features
        logger.info("Computing sorted features...")
        X_features_sorted, X_features_sorted_indices = self.sort_features(X_feature)
        logger.info("Finished computing sorted features!")

        return Dataset(X, y, image_features, X_integral, X_feature, X_features_sorted, X_features_sorted_indices)
    
    def prepare_data(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
        self.train_ds = self._prepare_data(X_train, y_train, "train")
        self.valid_ds, self.test_ds = None, None

        if X_valid is not None and y_valid is not None:
            self.valid_ds = self._prepare_data(X_valid, y_valid, "valid")
            assert self.valid_ds.image_features == self.train_ds.image_features
        if X_test is not None and y_test is not None:
            self.test_ds = self._prepare_data(X_test, y_test, "test")
            assert self.test_ds.image_features == self.train_ds.image_features 

    def save(self, out_path, train_name="train", valid_name="valid", test_name="test"):
        self.train_ds.save(out_path, train_name)
        if self.valid_ds is not None:
            self.valid_ds.save(out_path, valid_name)
        if self.test_ds is not None:
            self.test_ds.save(out_path, test_name)

    def load_data(self, path, train_name="train", valid_name="valid", test_name="test"):
        logger.info("Loading data from {}, with files: {}, {}, {}".format(path, train_name, valid_name, test_name))
        self.train_ds = Dataset.load(path, train_name)
        self.valid_ds = Dataset.load(path, valid_name)
        self.test_ds = Dataset.load(path, test_name)
        logger.info("Finished loading data!")

    @staticmethod
    def build_features(height, width):
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while (i + w) <= width:
                    j = 0
                    while (j + h) <= height:
                        # 2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if (i + 2 * w) <= width: # Horizontally Adjacent
                            features.append(ImageFeature([right], [immediate]))
                        bottom = RectangleRegion(i, j+h, w, h)
                        if (j + 2 * h) <= height: # Vertically Adjacent
                            features.append(ImageFeature([immediate], [bottom]))
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        # 3 rectangle features
                        if (i + 3 * w) <= width: # Horizontally Adjacent
                            features.append(ImageFeature([right], [right_2, immediate]))
                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if (j + 3 * h) <= height: # Vertically Adjacent
                            features.append(ImageFeature([bottom], [bottom_2, immediate]))
                        # 4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if (i + 2 * w) <= width and (j + 2 * h) <= height:
                            features.append(ImageFeature([right, bottom], [immediate, bottom_right]))
                        j += 1
                    i += 1
        return features

def prepare_training_data():
    X_train, y_train = pickle_load('data/train.pkl')
    X_valid, y_valid = pickle_load('data/valid.pkl')
    X_test, y_test = pickle_load('data/test.pkl')

    trainer = Trainer(mp_pool=Pool(8))
    trainer.prepare_data(X_train, y_train, X_valid, y_valid, X_test, y_test)
    trainer.save('data2')

def train_boosted(fast_experiment=True):
    if fast_experiment:
        # use a very small data set for fast experiment
        init_logger("test.log")
        trainer = Trainer(mp_pool=Pool(8))
        trainer.load_data('data', train_name="test", valid_name="valid", test_name="")
        trainer.train_boosted(n_weak=2000, model_save_path="models_test")
    else:
        init_logger("strong_classifier7.log")
        trainer = Trainer(mp_pool=Pool(8))
        trainer.load_data('data')
        trainer.train_boosted(n_weak=2000, model_save_path="models7")

def debug_prepare_training_data():
    X, y = pickle_load('data/valid.pkl')
    #X, y = X[:10], y[:10]
    trainer1 = Trainer(mp_pool=None)
    trainer1.prepare_data(X, y)
    trainer2 = Trainer(mp_pool=Pool(8))
    trainer2.prepare_data(X, y)
    print(np.sum(np.abs(trainer1.train_ds.X_features - trainer2.train_ds.X_features)))

def debug_build_features(count_only=False):
    all_features = Trainer.build_features(2, 2)
    print(len(all_features))
    if count_only:
        return
    for feature in all_features:
        print(feature)
    pickle_save(all_features, "all_features.pkl")
    feature_reloaded = pickle_load('all_features.pkl')
    assert feature_reloaded == all_features

def debug_test_set():
    clf = pickle_load(os.path.join('models6', 'strong_classifier_276.pkl'))
    trainer = Trainer(mp_pool=Pool(8))
    trainer.load_data('data')
    print("Strong classifier test metrics:")
    predictions = clf.classify_batch(trainer.test_ds.X_integral)
    print(Metrics(predictions, trainer.test_ds.y))

if __name__ == "__main__":
    #debug_prepare_training_data()
    #debug_build_features()
    debug_test_set()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_training_data()
    elif args.mode == "boost":
        train_boosted(fast_experiment=False)
    elif args.mode == "fast":
        train_boosted(fast_experiment=True)
