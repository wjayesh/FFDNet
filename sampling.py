from math import floor
import numpy as np

def downsample(dataset):
    """Code to downsample a dataset to provide as input.
    """
    ds_ds = np.zeros([len(dataset), 5, 25, 8])

    for x in range(5):
        for c in range(8):    
            for y in range(25):
                if y >= 25 or 2*y + (c%2) >= 25:
                    continue
                if x >= 5 or 2*x + (c%2) >= 5:
                    continue
                ds_ds[:, x, y, c] = dataset[:, 2*x + (c%2), 2*y + (c%2), floor(c/4)]
    return ds_ds

def upsample(dataset: np.ndarray) -> np.ndarray:
    """Code to upsample a dataset coming from the output of the internal
    DnCNN network.
    """
    us_ds = np.zeros([len(dataset), 10, 51, 2])

    for c in range(8):
        for x in range(5):
            for y in range(25):
                us_ds[:, 2*x + (c%2), 2*y + (c%2), floor(c/4)] = dataset[:, x, y, c]
    return us_ds