import numpy as np
import multiprocessing as mp

def matching_pixels(y_true, y_pred):
    return np.sum(y_true == y_pred)

def matching_pixels_subset_min(y_true_set, y_pred):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap(matching_pixels, zip(y_true_set, y_pred))
    return min(results)