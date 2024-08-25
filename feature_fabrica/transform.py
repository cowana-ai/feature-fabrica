# transform.py
import numpy as np


def sum_fn(iterable):
    return sum(iterable)


def to_str(data):
    return f"{data}"


def scale_feature(data, factor):
    return data * factor


def log_transform(data):
    return np.log(data + 1)
