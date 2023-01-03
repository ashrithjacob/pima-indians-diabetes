"""
functions:
sk_api(); dmatrix()
    - input args: number_of_trees, scale_pos_weight, X_train, X_test, Y_train, Y_test
    - RETURN: trained model

"""
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK
from data import *

def sk_api( param, number_of_trees, scale_pos_weight, X_train, X_test, Y_train, Y_test):

    # XGB classifier
    boosted_tree = xgb.XGBClassifier(
        colsample_bytree=int(param["colsample_bytree"]),
        eta=param["eta"],
        eval_metric="auc",
        gamma=param["gamma"],
        max_depth=int(param["max_depth"]),
        min_child_weight=int(param["min_child_weight"]),
        n_estimators=number_of_trees,
        objective="binary:logistic",
        reg_alpha=int(param["reg_alpha"]),
        reg_lambda=param["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        subsample=param["subsample"],
    )

    evaluation = [(X_train, Y_train), (X_test, Y_test)]
    boosted_tree.fit(
        X_train,
        Y_train,
        eval_set=evaluation,
        verbose=False,
    )
    Y_pred = boosted_tree.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred>0.5)
    return {"loss": -accuracy, "status": STATUS_OK}


def d_matrix(
    param, number_of_trees, scale_pos_weight, missing_val, X_train, X_test, Y_train, Y_test
):
    num_round = number_of_trees

    D_train = xgb.DMatrix(X_train, Y_train, missing=missing_val)
    D_tests = xgb.DMatrix(X_test, Y_test)
    watchlist_train = [(D_train, "train")]
    watchlist_tests = [(D_tests, "tests")]

    print("loading data end, start to boost trees")
    boosted_tree = xgb.train(
        param,
        D_train,
        num_round,
        evals=watchlist_train,
        verbose_eval=100,
    )

    xgmat = xgb.DMatrix(X_test, missing=missing_val)
    Y_pred = boosted_tree.predict(xgmat, strict_shape=True)
    accuracy = accuracy_score(Y_test, Y_pred>0.5)

    return {"loss": -accuracy, "status": STATUS_OK}
