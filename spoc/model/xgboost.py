# coding: utf-8


import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


def xgb_train(X_train, X_test, y_train, y_test, task_type, best_params, feval, verbose_eval=50):

    params = {
        "booster" : "gbtree",        
        "tree_method": 'auto',
    }

    if task_type == "binary_classification":
        params["objective"] = "binary:logistic" 
        params["eval_metric"] = "auc"
        
    elif task_type == "regression":
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"

    params["max_depth"] = int(round(best_params["params"]["max_depth"]))
    params['learning_rate'] = best_params["params"]["learning_rate"]
    params['colsample_bytree'] = max(min(best_params["params"]["colsample_bytree"], 1), 0)
    params['subsample'] = max(min(best_params["params"]["subsample"], 1), 0)
    params['eta'] = best_params["params"]["eta"]
    params['gamma'] = best_params["params"]["gamma"]  

    train_set = xgb.DMatrix(X_train, label=y_train)

    model = xgb.train(
        params,
        train_set,
        num_boost_round=int(best_params["params"]["num_boost_round"]),
        verbose_eval=verbose_eval,
        feval=feval,
    )
    
    # prediction
    dtrain = xgb.DMatrix(X_train)
    dtest = xgb.DMatrix(X_test)

    if task_type == "binary_classification":
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        print(y_test_pred)
        roc_auc_train = roc_auc_score(y_train, y_train_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_pred)

        criteria = {
            'roc_auc_train': roc_auc_train,
            'roc_auc_test': roc_auc_test,
        }

    elif task_type == "regression":
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        rmse_train, mae_train, r2_train = metric_cal(y_train, y_train_pred, precise=4)
        rmse_test, mae_test, r2_test = metric_cal(y_test, y_test_pred, precise=4)
        
        criteria = {
            'rmse_train': rmse_train,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
        }

    return criteria


def bayesion_opt_xgb(X_train, y_train, task_type, init_iter, n_iters, feval, criterion_col, pds='default', random_state=42, seed=42):

    if pds == 'default':
        pds = {
            'num_boost_round': (200, 2000),
            'max_depth': (3, 10),
            'learning_rate': (0.005, 0.3),
            'colsample_bytree': (0.5, 1),
            'subsample': (0.6, 1),
            'eta': (0.001, 0.1),
            'gamma': (0, 25),
        }

    # Objective Function
    def xgb_cv(num_boost_round, max_depth, learning_rate, colsample_bytree, subsample, eta, gamma):

        train_set  = xgb.DMatrix(data=X_train, label=y_train)

        params = {
            "booster" : "gbtree",            
            "tree_method": 'auto',
        }  

        if task_type == "binary_classification":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "auc"
            
        elif task_type == "regression":
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
        
        params["max_depth"] = int(round(max_depth))
        params['learning_rate'] = learning_rate
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params['eta'] = eta
        params['gamma'] = gamma
        
        cv_results = xgb.cv(
            params, 
            train_set, 
            num_boost_round=int(round(num_boost_round)), 
            nfold=5, 
            stratified=False, 
            folds=None, 
            metrics=(), 
            obj=None, 
            feval=feval, 
            maximize=None, 
            fpreproc=None, 
            as_pandas=True, 
            show_stdv=True, 
            seed=seed, 
            callbacks=None, 
            shuffle=True,
        )
        
        if criterion_col == "rmse_test_ave":
            result = -np.min(cv_results["test-mse-mean"])
        elif criterion_col == "mae_test_ave":
            result = -np.min(cv_results["test-mae-mean"])
        elif criterion_col == "r2_test_ave":
            result = np.max(cv_results["test-r2-mean"])
        elif criterion_col == "roc_auc_test_ave":
            result = np.max(cv_results["test-roc_auc-mean"])

        return result

    optimizer = BayesianOptimization(xgb_cv, pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    return optimizer.max


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


def xgb_mse_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mse', mean_squared_error(labels, preds)


def xgb_mae_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(labels, preds)


def xgb_roc_auc_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc', roc_auc_score(labels, preds)


def metric_cal(y_true, y_pred, precise=3):
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
        feval = xgb_mse_score 
    elif criterion_col == "mae_test_ave":
        ascending = True
        feval = xgb_mae_score
    elif criterion_col == "r2_test_ave":
        ascending = False
        feval = xgb_r2_score
    elif criterion_col == "roc_auc_test_ave":
        ascending = False
        feval = xgb_roc_auc_score
        
    return feval, ascending