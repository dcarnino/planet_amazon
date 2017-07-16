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




#==============================================
#                   Main
#==============================================
if __name__ == '__main__':

    y_pred, y_true = [], []

    for fold_id in range(5):

        df_val = pd.read_csv("../data/planet_amazon/val%d.csv"%fold_id)

        with open("../data/planet_amazon/inceptionv3_preds%d.npy"%fold_id, "rb") as iOF:
            y_pred_fold = np.load(iOF)

        y_pred_fold = np.mean(y_pred_fold, axis=0)
        y_pred.append(y_pred_fold)

        y_true_fold = df_val.iloc[:,2:].values
        y_true.append(y_true_fold)

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    y_pred[y_pred > 0.8] = 1
    y_pred[y_pred < 0.8] = 0

    print(y_true.shape)
    print(y_pred.shape)

    print(y_true[:3,:])
    print(y_pred[:3,:])

    print("Fbeta score: ", fbeta_score(y_true, y_pred, 2, average='samples'))
