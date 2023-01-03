"""
Created on Tuesday December 13th 10:36:40 2022
@author: ashrith

DATASET: pima-indians-diabetes dataset
Information on dataset: SEE README

'black' package used as linter

split into train and test:
see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
or : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

About Hyperopt:
'fmin' optimizes 'objective'(function) over a 'space'(dict)
'objective': any defined function
'space': dict of hyperparam space to optimize variable in
'fmin' : hyperopt function to minimize return val of objective function
        for values in the space.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from data import *
from pred import *

"""
Editable parameters by user:
"""
# path to dataset
PATH = "dataset/pima-indians-diabetes-withcolnames.csv"
# fraction of data used as test split
test_split = 0.1
# value for missing information
missing_val = np.NaN
# also referred to as n_estimators
number_of_trees = 2000
"""
End of editable section
"""

# main only execs if this is script is executed
if __name__ == "__main__":
    # read csv
    df = pd.read_csv(PATH)
    # Set dependant and independant variables
    ncols = df.shape[1]
    X = df.iloc[:, 0 : ncols - 1]
    Y = df.iloc[:, ncols - 1]

    # clean data
    clean(df, missing_val)

    # display data stats
    display(df, ncols, X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_split, random_state=42, shuffle=True, stratify=Y
    )
    scale_pos_weight = sum(Y_train[:] == 0) / sum(Y_train[:] == 1)

    # hyperparam_tuning(ps_aut)
    space = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
        "eval_metric": "auc",
        "gamma": hp.uniform("gamma", 0, 9),
        "max_depth": hp.randint("max_depth", 3, 18),
        "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "n_estimators": number_of_trees,
        "objective": "binary:logistic",
        "reg_alpha": hp.quniform("reg_alpha", 0, 180, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "scale_pos_weight": scale_pos_weight,
        "subsample": hp.quniform("subsample", 0.025, 1, 0.025),
    }

# define an objective function


def objective_skapi(space):
    clf = xgb.XGBClassifier(
        colsample_bytree=int(space["colsample_bytree"]),
        eta=space["eta"],
        eval_metric=space["eval_metric"],
        gamma=space["gamma"],
        max_depth=int(space["max_depth"]),
        min_child_weight=int(space["min_child_weight"]),
        n_estimators=space["n_estimators"],
        objective=space["objective"],
        reg_alpha=int(space["reg_alpha"]),
        reg_lambda=space["reg_lambda"],
        scale_pos_weight=space["scale_pos_weight"],
        subsample=space["subsample"],
    )
    evaluation = [(X_train, Y_train), (X_test, Y_test)]
    clf.fit(
        X_train,
        Y_train,
        eval_set=evaluation,
        verbose=False,
    )
    Y_pred = clf.predict(X_test)
    print("Y_pred", Y_pred[:])
    accuracy = accuracy_score(Y_test, Y_pred)
    print("SCORE:", accuracy)
    return {"loss": -accuracy, "status": STATUS_OK}


# define an objective function
def objective_dmatrix(space):
    params = list(space.items())
    num_round = number_of_trees

    D_train = xgb.DMatrix(X_train, Y_train, missing=missing_val)
    D_tests = xgb.DMatrix(X_test, Y_test)
    watchlist_train = [(D_train, "train")]
    watchlist_tests = [(D_tests, "tests")]

    print("loading data end, start to boost trees")
    boosted_tree = xgb.train(
        params,
        D_train,
        num_round,
        evals=watchlist_train,
        verbose_eval=100,
    )

    xgmat = xgb.DMatrix(X_test, missing=missing_val)
    Y_pred = boosted_tree.predict(xgmat, strict_shape=True)
    accuracy = accuracy_score(Y_test, Y_pred > 0.5)

    return {"loss": -accuracy, "status": STATUS_OK}


trials = Trials()

if sys.argv[1] == "SKAPI":
    best_hyperparams = fmin(
        fn=objective_skapi, space=space, algo=tpe.suggest, max_evals=100, trials=trials
    )
elif sys.argv[1] == "DMATRIX":
    best_hyperparams = fmin(
        fn=objective_dmatrix,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )
else:
    print(
        "INCORRECT METHOD PASSED- ONLY FOLLOWING METHODS PERMITTED:\n 1. SKAPI\n 2. DMATRIX"
    )
    exit(1)
print("The best hyperparameters are : ", "\n")
print(best_hyperparams)
