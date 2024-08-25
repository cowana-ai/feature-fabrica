# transform.py
import numpy as np


def scale_feature(data, factor):
    return data * factor


def log_transform(data):
    return np.log(data + 1)
