import numpy as np
import multiprocessing as mp
from tqdm import tqdm
def matching_pixels(y_true, y_pred):
    return np.sum((y_true == y_pred).astype(np.float32))

def matching_pixels_bridge(combined):
    y_true, y_pred = combined
    return matching_pixels(y_true, y_pred)

def matching_pixels_subset_min(y_true_set, y_pred):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap(matching_pixels_bridge, [(y_true, y_pred) for y_true in y_true_set])
        
    return min(results)

def matching_pixels_subset_max(y_true_set, y_pred):
    #arr = [(y_true, y_pred) for y_true in y_true_set]
    #with mp.Pool(mp.cpu_count()) as pool:
        #results = pool.imap(matching_pixels_bridge, arr)
    res = []
    for y_true in tqdm(y_true_set):
        res.append(matching_pixels(y_true, y_pred))
    return max(res)
