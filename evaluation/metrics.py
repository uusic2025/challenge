import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


def compute_dsc(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    
    intersection = np.sum(y_true * y_pred)
    dsc = 2.0 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-6)
    return dsc


def compute_nsd(y_true, y_pred, tolerance=1):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    boundary_true = find_boundaries(y_true, mode='inner')
    boundary_pred = find_boundaries(y_pred, mode='inner')

    distance_true = distance_transform_edt(1 - boundary_true)
    distance_pred = distance_transform_edt(1 - boundary_pred)

    true_in_pred = (boundary_true & (distance_pred <= tolerance)).sum()
    pred_in_true = (boundary_pred & (distance_true <= tolerance)).sum()

    nsd = (true_in_pred + pred_in_true) / (boundary_true.sum() + boundary_pred.sum() + 1e-6)
    return nsd
