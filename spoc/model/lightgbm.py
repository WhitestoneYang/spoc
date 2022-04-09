# coding: utf-8

import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def lgb_train(X_train, X_test, y_train, y_test, best_params, feval, seed=42, verbose_eval=50, bagging_seed=42, verbosity=-1, task='binary_classification'):
    
    if task == 'binary_classification':
        objective = 'binary'
    elif task == 'regression':
        objective = 'regression'
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': objective,
        "bagging_fraction": best_params["params"]["bagging_fraction"],
        "feature_fraction": best_params["params"]["feature_fraction"],
        "max_depth": int(best_params["params"]["max_depth"]),
        "min_child_weight": best_params["params"]["min_child_weight"],
        "min_split_gain": best_params["params"]["min_split_gain"],
        "num_leaves": int(best_params["params"]["min_child_weight"]),
        "learning_rate": best_params["params"]["learning_rate"],
        "bagging_seed": bagging_seed,
        "verbosity": verbosity,
        "seed": seed,
    }

    lg_train = lgb.Dataset(X_train, label=y_train)

    model = lgb.train(
        params,
        lg_train,
        num_boost_round=int(best_params["params"]["num_boost_round"]),
        verbose_eval=verbose_eval,
        feval=feval,
    )

    # prediction & critera calculation
    if task == 'binary_classification':
        y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        roc_auc_train = roc_auc_score(y_train, y_train_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_pred)
        criteria = {
            'roc_auc_train': roc_auc_train,
            'roc_auc_test': roc_auc_test,
        }
        
    elif task == 'regression':
        y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        criteria = {
            'rmse_train': rmse_train,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
        }        

    return criteria


def bayesopt_lgb(X_train, y_train, init_iter, n_iters, feval, criterion_col, pds='default', random_state=42, seed=42, task='binary_classification', verbosity=-1):

    if pds == 'default':
        pds = {
            'num_boost_round': (200, 2000),
            'num_leaves': (10, 255),
            'feature_fraction': (0.1, 0.9),
            'bagging_fraction': (0.5, 1),
            'max_depth': (4, 16),
            'min_split_gain': (0.001, 0.1),
            'min_child_weight': (4, 128),
            'learning_rate': (0.002, 0.05),
        }      

    # Objective Function
    def lgb_cv(num_boost_round, num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight, learning_rate, nfold=5):

        train_set = lgb.Dataset(data=X_train, label=y_train)

        if task == 'binary_classification':
            objective = 'binary'
        elif task == 'regression':
            objective = 'regression'
        
        params = {
            'boosting_type': 'gbdt',
            'objective': objective,
            'verbose': verbosity,
        }
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['learning_rate'] = learning_rate

        cv_results = lgb.cv(
            params,
            train_set,
            num_boost_round=int(round(num_boost_round)),
            nfold=nfold,
            stratified=False,
            shuffle=True,
            feval=feval,
            seed=seed,
        )

        # critera
        if criterion_col == "rmse_test_ave":
            result = -np.min(cv_results["mse-mean"])
        elif criterion_col == "mae_test_ave":
            result = -np.min(cv_results["mae-mean"])
        elif criterion_col == "r2_test_ave":
            result = np.max(cv_results["r2-mean"])
        elif criterion_col == "roc_auc_test_ave":
            result = np.max(cv_results["roc_auc-mean"])

        return result
    
    optimizer = BayesianOptimization(lgb_cv, pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    return optimizer.max


def lgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds), True


def lgb_mse_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mse', mean_squared_error(labels, preds), False


def lgb_mae_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(labels, preds), False


def lgb_roc_auc_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc', roc_auc_score(labels, preds), True


def metric_cal(y_true, y_pred, precise=3):
    """ rmse, mae, r2 """
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), precise)
    mae = round(mean_absolute_error(y_true, y_pred), precise)
    r2 = round(r2_score(y_true, y_pred), precise)
    return rmse, mae, r2


def feval_value(criterion_col):
    """To determine the score function.

    Args:
        criterion_col (str): criterion_type

    Returns:
        (func, bool): (feval, ascending)
    """
    if criterion_col == "rmse_test_ave":
        ascending = True
        feval = lgb_mse_score 
    elif criterion_col == "mae_test_ave":
        ascending = True
        feval = lgb_mae_score
    elif criterion_col == "r2_test_ave":
        ascending = False
        feval = lgb_r2_score
    elif criterion_col == "roc_auc_test_ave":
        ascending = False
        feval = lgb_roc_auc_score
        
    return feval, ascending
