import numpy as np

def fm_measure(y_true, y_pred):
    """
    Example input:
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    """

    tp = np.sum(y_true & y_pred)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(y_true & ~y_pred)
    print(tp / (np.sqrt(tp + fp) * np.sqrt(tp + fn)))
    return tp / (np.sqrt(tp + fp) * np.sqrt(tp + fn))

