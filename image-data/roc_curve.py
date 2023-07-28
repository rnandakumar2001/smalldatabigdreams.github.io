import numpy as np
import matplotlib.pyplot as plt

def roc_curve(y_true, y_score):
    """ 
     Ex input:
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    """
    fpr = []
    tpr = []
    for threshold in np.arange(0, 1.01, 0.01):
        tp = np.sum(y_true & (y_score >= threshold))
        fp = np.sum(~y_true & (y_score >= threshold))
        fpr.append(fp / float(len(y_true) - np.sum(y_true)))
        tpr.append(tp / float(np.sum(y_true)))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    return fpr, tpr