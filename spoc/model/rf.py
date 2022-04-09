# coding: utf-8

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error


def rf_cls(X_train, X_test, y_train, y_test, n_estimators="default", criterion="default", max_features="default", max_depth="default", rnd=42):
    """Random forest classification method. 

    Args:
        X_train (np.array): X training set
        X_test (np.array): X test set
        y_train (np.array): y training set
        y_test (np.array): y test set
        n_estimators (str, optional): Defaults to "default".
        criterion (str, optional):  Defaults to "default".
        max_features (str, optional): Defaults to "default".
        max_depth (str, optional): Defaults to "default".
        rnd (int, optional): Defaults to 42.

    Returns:
        (float, float): (roc_auc_train, roc_auc_test)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=max_features,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-2,
        random_state=rnd,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )

    model.fit(X_train, y_train)

    # criterion
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1].ravel()
    roc_auc_train = roc_auc_score(y_train, y_pred_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)
    
    criteria = {
                'roc_auc_train': roc_auc_train,
                'roc_auc_test': roc_auc_test,
            }

    return criteria


def rf_reg(X_train, X_test, y_train, y_test, n_estimators="default", criterion="default", max_features="default", max_depth="default", rnd=42):
    """random forest regression method. 

    Args:
        X_train (np.array): X training set
        X_test (np.array): X test set
        y_train (np.array): y training set
        y_test (np.array): y test set
        n_estimators (str, optional): Defaults to "default".
        criterion (str, optional):  Defaults to "default".
        max_features (str, optional): Defaults to "default".
        max_depth (str, optional): Defaults to "default".
        rnd (int, optional): Defaults to 42.

    Returns:
        (float, float): 
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=max_features,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=rnd,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )

    model.fit(X_train, y_train)

    # criterion
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    criteria = {
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'r2_train': r2_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
    }
    
    return criteria
