"""
    Name:           test_model_inceptionv3.py
    Created:        16/7/2017
    Description:    Fine-tune inception v3 for Planet Amazon.
"""
#==============================================
#                   Modules
#==============================================
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
#==============================================
#                   Files
#==============================================


#==============================================
#                   Functions
#==============================================
def optimise_f2_thresholds(y, p, resolution=100, verbose=1):

    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.5]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose >= 1:
            print(i, best_i2, best_score)

    return x



#==============================================
#                   Main
#==============================================
if __name__ == '__main__':

    y_pred, y_true = [], []

    for fold_id in range(5):

        df_val = pd.read_csv("../data/planet_amazon/val%d.csv"%fold_id)

        with open("../data/planet_amazon/inceptionv3_preds%d.npy"%fold_id, "rb") as iOF:
            y_pred_fold = np.load(iOF)
        with open("../data/planet_amazon/inceptionv3_trues%d.npy"%fold_id, "rb") as iOF:
            y_true_fold = np.load(iOF)

        #y_pred_fold = np.mean(y_pred_fold, axis=0)
        y_pred_fold = y_pred_fold[0,...]
        y_pred.append(y_pred_fold)

        #y_true_fold = np.mean(y_true_fold, axis=0)
        y_true_fold = y_true_fold[0,...]
        y_true.append(y_true_fold)

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    #f2_threshs = optimise_f2_thresholds(y_true, y_pred, resolution=100)
    f2_threshs = [0.5]*17

    y_pred2 = np.zeros_like(y_pred)
    for i in range(17):
        y_pred2[:, i] = (y_pred[:, i] > f2_threshs[i]).astype(np.int)

    print(y_true.shape)
    print(y_pred2.shape)

    print(y_true[:3,:])
    print(y_pred2[:3,:])

    print("Fbeta score: ", fbeta_score(y_true, y_pred2, 2, average='samples'))
