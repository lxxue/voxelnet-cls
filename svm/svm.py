import numpy as np
from sklearn.svm import SVC
import logging
import os

def main():
    # for m10
    train_features = np.load("../data/svm/m10_train_features.npy")
    train_labels = np.load("../data/svm/m10_train_labels.npy")
    val_features = np.load("../data/svm/m10_val_features.npy")
    val_labels = np.load("../data/svm/m10_val_labels.npy")

    logger = init_logger("m10_log")
    m10_svm = SVC(kernel='linear')
    m10_svm.fit(train_features, train_labels)
    train_pred = m10_svm.predict(train_features)
    val_pred = m10_svm.predict(val_features)
    logger.info(str(np.sum(train_pred == train_labels) / len(train_labels)))
    logger.info(str(np.sum(val_pred == val_labels) / len(val_labels)))

    logger_few = init_logger("few_m10_log/")
    for i in range(10):
        train_features_i, train_labels_i = random_sample(train_features, train_labels, 10, 9) 
        few_m10_svm = SVC(kernel='linear')
        few_m10_svm.fit(train_features_i, train_labels_i)
        train_pred_i = few_m10_svm.predict(train_features_i)
        val_pred = few_m10_svm.predict(val_features)
        logger_few.info("Experiment {}:".format(i+1))
        logger_few.info(str(np.sum(train_pred_i == train_labels_i) / len(train_labels_i)))
        logger_few.info(str(np.sum(val_pred == val_labels) / len(val_labels)))


def init_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    log_fname = os.path.join(log_dir, "log.txt")
    fh = logging.FileHandler(log_fname)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def random_sample(features, labels, num_classes, num_per_class):
    sampled_features = []
    sampled_labels = []
    for i in range(num_classes):
        idx = np.squeeze(labels == i)
        features_i = features[idx]
        labels_i = labels[idx]
        assert np.all(labels_i == np.mean(labels_i, dtype=np.int))
        randidx = np.random.permutation(len(features_i))[:num_per_class]
        features_i = features_i[randidx]
        labels_i = labels_i[randidx]
        sampled_features.append(features_i)
        sampled_labels.append(labels_i)
    
    sampled_features = np.concatenate(sampled_features)
    sampled_labels = np.concatenate(sampled_labels)
    print(sampled_features.shape)
    print(sampled_labels.shape)

    return sampled_features, sampled_labels


if __name__ == "__main__":
    main()
