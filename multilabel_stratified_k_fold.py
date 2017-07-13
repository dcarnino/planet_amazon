import numpy as np
import pandas as pd
import sys

def proba_mass_split(y, n_folds=5, shuffle=True, verbose=1):
    y_index = np.array(range(y_train.shape[0]))
    if shuffle:
        np.random.shuffle(y_index)
        y = y[y_index]
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((n_folds, classes), dtype='float')
    for _ in range(n_folds):
        index_list.append([])
    for i, j in enumerate(y_index):
        if i < n_folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(j)
    if verbose >= 2:
        print("Fold distributions are")
        print(fold_dist)
    return index_list


if __name__ == '__main__':

    # Import df
    df_train = pd.read_csv("../data/planet_amazon/train_v2.csv")

    # transform list of str to columns of binary labels
    tag_set = sorted(set([tag for tag_list in df_train.tags for tag in tag_list.split(' ')]))
    for tag in tag_set:
        df_train[tag] = df_train.tags.apply(lambda x: 1 if tag in x else 0)
    y_train = df_train.iloc[:,2:].values

    # stratified k-fold
    offset_folds = int(sys.argv[1])
    n_folds = 5
    ix_folds = proba_mass_split(y_train, n_folds=n_folds)

    # write to file
    for fold_id in range(n_folds):
        ix_train = np.array([itm for lst in ix_folds[:fold_id] for itm in lst] + [itm for lst in ix_folds[fold_id+1:] for itm in lst])
        ix_val = np.array(ix_folds[fold_id])
        df_train.iloc[ix_train,:].to_csv("../data/planet_amazon/train%d.csv"%(fold_id+offset_folds), index=False)
        df_train.iloc[ix_val,:].to_csv("../data/planet_amazon/val%d.csv"%(fold_id+offset_folds), index=False)
