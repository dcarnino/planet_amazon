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
from sklearn.metrics import fbeta_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import kurtosis, skew
import gzip
#==============================================
#                   Files
#==============================================
from xgboost_ensembling import XGBClassifier_ensembling


#==============================================
#                   Functions
#==============================================
def optimise_f2_thresholds(y, p, resolution=100, bmin=0., bmax=1., verbose=1):

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
        for i2 in np.linspace(bmin, bmax, resolution):
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

    mean_only = False
    cross_validate = True

    y_true = []
    y_pred_mean, y_pred_median, y_pred_std, y_pred_min, y_pred_max, y_pred_skew, y_pred_kurtosis = [], [], [], [], [], [], []

    for fold_id in range(5):

        df_val = pd.read_csv("../data/planet_amazon/val%d.csv"%fold_id)

        with open("../data/planet_amazon/inceptionv3_preds%d.npy"%fold_id, "rb") as iOF:
            y_pred_fold = np.load(iOF)
        with open("../data/planet_amazon/inceptionv3_trues%d.npy"%fold_id, "rb") as iOF:
            y_true_fold = np.load(iOF)

        y_pred_fold_mean = np.mean(y_pred_fold, axis=0)
        y_pred_fold_median = np.median(y_pred_fold, axis=0)
        y_pred_fold_std = np.std(y_pred_fold, axis=0)
        y_pred_fold_min = np.min(y_pred_fold, axis=0)
        y_pred_fold_max = np.max(y_pred_fold, axis=0)
        y_pred_fold_skew = skew(y_pred_fold, axis=0, nan_policy='omit')
        y_pred_fold_skew[~np.isfinite(y_pred_fold_skew)] = 0.
        y_pred_fold_kurtosis = kurtosis(y_pred_fold, axis=0, nan_policy='omit')
        y_pred_fold_kurtosis[~np.isfinite(y_pred_fold_kurtosis)] = 0.
        y_pred_mean.append(y_pred_fold_mean)
        y_pred_median.append(y_pred_fold_median)
        y_pred_std.append(y_pred_fold_std)
        y_pred_min.append(y_pred_fold_min)
        y_pred_max.append(y_pred_fold_max)
        y_pred_skew.append(y_pred_fold_skew)
        y_pred_kurtosis.append(y_pred_fold_kurtosis)

        y_true_fold = np.mean(y_true_fold, axis=0)
        y_true.append(y_true_fold)

    y_pred_mean = np.vstack(y_pred_mean)
    y_pred_median = np.vstack(y_pred_median)
    y_pred_std = np.vstack(y_pred_std)
    y_pred_min = np.vstack(y_pred_min)
    y_pred_max = np.vstack(y_pred_max)
    y_pred_skew = np.vstack(y_pred_skew)
    y_pred_kurtosis = np.vstack(y_pred_kurtosis)

    y_pred = np.array([y_pred_mean, y_pred_median, y_pred_std, y_pred_min, y_pred_max, y_pred_skew, y_pred_kurtosis])
    y_true = np.vstack(y_true)

    if mean_only:

        f2_threshs = optimise_f2_thresholds(y_true, y_pred_mean, resolution=100)
        #f2_threshs = [0.5]*17

        with open("../data/planet_amazon/optimized_thresholds_inceptionv3.txt", "w") as iOF:
            iOF.writelines([str(thresh)+"\n" for thresh in f2_threshs])

        y_pred2 = np.zeros_like(y_pred_mean)
        for i in range(17):
            y_pred2[:, i] = (y_pred_mean[:, i] > f2_threshs[i]).astype(np.int)

        print(y_true.shape)
        print(y_pred2.shape)

        print(y_true[:3,:])
        print(y_pred2[:3,:])

        print("Fbeta score: ", fbeta_score(y_true, y_pred2, 2, average='samples'))

    else:

        y_pred_xgb = np.zeros_like(y_true)

        for ix_feat in range(17):

            if cross_validate:

                y_pred_feat = y_pred[..., ix_feat].T
                y_true_feat = y_true[..., ix_feat]

                n_folds = 5
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

                for fold_cnt, (train_index, test_index) in enumerate(cv.split(y_pred_feat, y_true_feat)):

                    print("XGB feat %d/%d, fold %d/%d..."%(ix_feat+1,17,fold_cnt+1,n_folds))

                    XX_train, XX_test = y_pred_feat[train_index], y_pred_feat[test_index]
                    yy_train, yy_test = y_true_feat[train_index], y_true_feat[test_index]

                    clf = XGBClassifier_ensembling(n_folds=20, early_stopping_rounds=10,
                                                   max_depth=5, learning_rate=0.02,
                                                   objective='binary:logistic', nthread=28,
                                                   min_child_weight=4, subsample=0.7)

                    clf.fit(XX_train, yy_train)

                    yy_pred = clf.predict_proba(XX_test)
                    print(f1_score(yy_test, np.round(yy_pred), average='micro'))

                    y_pred_xgb[test_index, ix_feat] = yy_pred

            else:

                y_pred_feat = y_pred[..., ix_feat]
                y_true_feat = y_true[..., ix_feat]

                clf = XGBClassifier_ensembling(n_folds=20, early_stopping_rounds=10,
                                               max_depth=5, learning_rate=0.02,
                                               objective='binary:logistic', nthread=28,
                                               min_child_weight=4, subsample=0.7)

                clf.fit(y_pred_feat, y_true_feat)

                with gzip.open("../data/planet_amazon/models/xgb_class%d.gzip", "wb") as iOF:
                    iOF.dump(clf)


        if cross_validate:

            f2_threshs = optimise_f2_thresholds(y_true, y_pred_xgb, bmin=0.4, bmax=0.51, resolution=100)
            #f2_threshs = [0.5]*17

            with open("../data/planet_amazon/optimized_thresholds_xgb.txt", "w") as iOF:
                iOF.writelines([str(thresh)+"\n" for thresh in f2_threshs])

            y_pred2 = np.zeros_like(y_pred_xgb)
            for i in range(17):
                y_pred2[:, i] = (y_pred_xgb[:, i] > f2_threshs[i]).astype(np.int)

            print(y_true.shape)
            print(y_pred2.shape)

            print(y_true[:3,:])
            print(y_pred2[:3,:])

            print("Fbeta score: ", fbeta_score(y_true, y_pred2, 2, average='samples'))
