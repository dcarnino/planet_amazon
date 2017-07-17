from xgboost import XGBRegressor, XGBClassifier
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats
import numpy as np



class XGBRegressor_ensembling(BaseEstimator, RegressorMixin):


    def __init__(self, n_folds=5, early_stopping_rounds=10, eval_metric=metrics.r2_score, greater_is_better=True, predict_median=False, prior=None,
                 max_depth=3, learning_rate=0.1, n_estimators=100000, silent=True,
                 objective='reg:linear', nthread=None,
                 gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, base_score=0.5, missing=None):

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing

        self.n_folds = n_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.predict_median = predict_median
        self.prior = prior


    def custom_eval(self, y_pred, dtest):
        y_test = dtest.get_label()
        assert len(y_test) == len(y_pred)
        score = self.eval_metric(y_test, y_pred)
        if self.greater_is_better:
            score = -score
        return 'custom_metric', score


    def fit(self, X, y, verbose=False):

        X, y = check_X_y(X, y)

        self.estimator_list_ = [ XGBRegressor(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                      n_estimators=self.n_estimators, silent=self.silent,
                                      objective=self.objective, nthread=self.nthread,
                                      gamma=self.gamma, min_child_weight=self.min_child_weight, max_delta_step=self.max_delta_step,
                                      subsample=self.subsample, colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bytree,
                                      reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
                                      base_score=self.base_score, missing=self.missing) for fold in range(self.n_folds)]

        cv = model_selection.KFold(n_splits=self.n_folds, shuffle=True)

        for fold_cnt, (train_index, test_index) in enumerate(cv.split(X)):

            if verbose >= 1: print("XGB fold %d/%d..."%(fold_cnt+1,self.n_folds))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.estimator_list_[fold_cnt].fit(X_train, y_train,
                                              eval_set=[(X_test, y_test)], eval_metric=(lambda y_pred, dtest: self.custom_eval(y_pred, dtest)),
                                              early_stopping_rounds=self.early_stopping_rounds, verbose=verbose)


    def predict(self, X):

        check_is_fitted(self, ['estimator_list_'])

        X = check_array(X)

        y_pred = [estimator.predict(X) for estimator in self.estimator_list_]
        if self.prior is not None:
            y_pred = np.average(y_pred, weights=np.array([self.prior(yp) for yp in y_pred]), axis=0)
        else:
            if self.predict_median:
                y_pred = np.median(y_pred, axis=0)
            else:
                y_pred = np.mean(y_pred, axis=0)

        return y_pred





class XGBClassifier_ensembling(BaseEstimator, ClassifierMixin):


    def __init__(self, n_folds=5, early_stopping_rounds=10,
                 eval_metric=(lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='micro')), greater_is_better=True,
                 max_depth=3, learning_rate=0.1, n_estimators=100000, silent=True,
                 objective='multi:softmax', nthread=None,
                 gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, base_score=0.5, missing=None):

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing

        self.n_folds = n_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better


    def custom_eval(self, y_pred, dtest):
        y_test = dtest.get_label()
        assert len(y_test) == len(y_pred)
        print(y_test)
        print(y_pred)
        print(y_test.shape)
        print(y_pred.shape)
        score = self.eval_metric(y_test, np.argmax(y_pred, axis=1))
        if self.greater_is_better:
            score = -score
        return 'custom_metric', score


    def fit(self, X, y, verbose=False):

        X, y = check_X_y(X, y)

        self.estimator_list_ = [ XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                      n_estimators=self.n_estimators, silent=self.silent,
                                      objective=self.objective, nthread=self.nthread,
                                      gamma=self.gamma, min_child_weight=self.min_child_weight, max_delta_step=self.max_delta_step,
                                      subsample=self.subsample, colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bytree,
                                      reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
                                      base_score=self.base_score, missing=self.missing) for fold in range(self.n_folds)]

        cv = model_selection.KFold(n_splits=self.n_folds, shuffle=True)

        for fold_cnt, (train_index, test_index) in enumerate(cv.split(X)):

            if verbose >= 1: print("XGB fold %d/%d..."%(fold_cnt+1,self.n_folds))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # remove new labels from validation set
            train_labels = set(y_train)
            test_labels = set(y_test)
            new_labels = test_labels - train_labels
            if len(new_labels) > 0:
                new_mask = np.array([yt in new_labels for yt in y_test])
                X_test = X_test[~new_mask]
                y_test = y_test[~new_mask]

            self.estimator_list_[fold_cnt].fit(X_train, y_train,
                                              eval_set=[(X_test, y_test)], eval_metric=(lambda y_pred, dtest: self.custom_eval(y_pred, dtest)),
                                              early_stopping_rounds=self.early_stopping_rounds, verbose=verbose)


    def predict(self, X):

        check_is_fitted(self, ['estimator_list_'])

        X = check_array(X)

        y_pred = np.array([estimator.predict(X) for estimator in self.estimator_list_])

        y_pred = stats.mode(y_pred)[1][0]

        return y_pred



    def predict_proba(self, X):

        check_is_fitted(self, ['estimator_list_'])

        X = check_array(X)

        y_pred = np.array([[p2 for p1, p2 in estimator.predict_proba(X)] for estimator in self.estimator_list_])

        y_pred = np.mean(y_pred, axis=0)

        return y_pred
