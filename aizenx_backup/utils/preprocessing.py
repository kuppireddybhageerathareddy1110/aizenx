import numpy as np


def normalize(data):

    data = np.array(data)

    return (data - data.min()) / (data.max() - data.min())