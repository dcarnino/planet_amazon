"""
    Name:           test_model_inceptionv3.py
    Created:        16/7/2017
    Description:    Fine-tune inception v3 for Planet Amazon.
"""
#==============================================
#                   Modules
#==============================================
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import kurtosis, skew, iqr, entropy
import gzip
import pickle
#==============================================
#                   Files
#==============================================
from xgboost_ensembling import XGBClassifier_ensembling


#==============================================
#                   Functions
#==============================================
def main(verbose=1):

    if sys.argv[1] != 'test':

        ##### Validation
        fold_id = int(sys.argv[1])

        ### Import data
        if verbose >= 1: print("Importing data (fold %d)..."%fold_id)

        df_val = pd.read_csv("../data/planet_amazon/val%d.csv"%(fold_id))
        df_train = pd.read_csv("../data/planet_amazon/train%d.csv"%(fold_id))
        df_feat = pd.read_csv("../data/planet_amazon/train_features.csv").rename(columns={"0": "image_name"})
        df_val = df_val.merge(df_feat, how='left', on="image_name")
        df_train = df_train.merge(df_feat, how='left', on="image_name")

        X_train = df_train.iloc[:,19:].values
        y_train = df_train.iloc[:,2:19].values
        X_val = df_val.iloc[:,19:].values
        y_val = df_val.iloc[:,2:19].values

        y_pred = np.zeros_like(y_val)

        ### Train, pickle, and infer
        if verbose >= 1: print("Training (fold %d)..."%fold_id)

        for ix_feat in range(17):

            if verbose >= 1: print("\tXGB feat %d/%d..."%(ix_feat+1,17))

            y_train_feat = y_train[:, ix_feat]

            clf = XGBClassifier_ensembling(n_folds=20, early_stopping_rounds=10,
                                           max_depth=7, learning_rate=0.02,
                                           objective='binary:logistic', nthread=28,
                                           min_child_weight=4, subsample=0.7, colsample_bytree=0.7)

            clf.fit(X_train, y_train_feat)

            y_pred_feat = clf.predict_proba(X_val)
            y_pred[:, ix_feat] = y_pred_feat

            with gzip.open("../data/planet_amazon/models/xgb_%d_class%d.gzip"%(fold_id,ix_feat), "wb") as iOF:
                pickle.dump(clf, iOF)

        y_pred = np.array([y_pred])
        y_val = np.array([y_val])

        if verbose >= 2:
            print(y_pred.shape)
            print(y_val.shape)
            print(y_pred[0,0,0])
            print(y_val[0,0,0])
            print("Fbeta score: ", fbeta_score(np.mean(y_val, axis=0), np.mean(y_pred, axis=0).round(), 2, average='samples'))

        ### Save preds
        if verbose >= 1: print("Saving preds (fold %d)..."%fold_id)
        with open("../data/planet_amazon/xgboost_preds%d.npy"%(fold_id), "wb") as iOF:
            np.save(iOF, y_pred)
        with open("../data/planet_amazon/xgboost_trues%d.npy"%(fold_id), "wb") as iOF:
            np.save(iOF, y_val)

    else:

        ### Import data
        if verbose >= 1: print("Importing data (fold %d)..."%fold_id)

        df_feat = pd.read_csv("../data/planet_amazon/test_features.csv").rename(columns={"0": "image_name"})
        image_ids = df_feat["image_name"].values
        X_test = df_feat.iloc[:,1:].values

        ##### Test
        y_pred_folds = []
        offset = 30
        for fold_id in range(offset,offset+20):

            ### Infer
            if verbose >= 1: print("Inferring (fold %d)..."%fold_id)

            y_pred = np.zeros((X_test.shape[0],17))

            for ix_feat in range(17):

                if verbose >= 1: print("\tXGB feat %d/%d..."%(ix_feat+1,17))

                with gzip.open("../data/planet_amazon/models/xgb_%d_class%d.gzip"%(fold_id,ix_feat), "rb") as iOF:
                    clf = pickle.load(iOF)

                y_pred_feat = clf.predict_proba(X_test)
                y_pred[:, ix_feat] = y_pred_feat

            y_pred_folds.append(y_pred)

        y_pred_folds = np.array(y_pred_folds)
        if verbose >= 2:
            print(y_pred_folds.shape)
            print(y_pred_folds[0,0,0])

        ### Save preds
        if verbose >= 1: print("Saving preds...")
        with open("../data/planet_amazon/xgboost_predstest.npy", "wb") as iOF:
            np.save(iOF, y_pred_folds)
        with open("../data/planet_amazon/xgboost_idstest.txt", "w") as iOF:
            iOF.writelines([image_id+"\n" for image_id in image_ids])






#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    main(2)
