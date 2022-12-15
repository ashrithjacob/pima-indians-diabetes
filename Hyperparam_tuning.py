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

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from data import *
from pred import *

"""
Editable parameters by user:
"""
PATH = "dataset/pima-indians-diabetes-withcol.csv"  # path to dataset
test_split = 0.1  # what fraction of total data is to be used as test split
missing_val = np.NaN
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
    "objective": "binary:logistic",
    "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 0, 9),
    "reg_alpha": hp.quniform("reg_alpha", 0, 180, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "subsample": hp.quniform("subsample", 0.025, 1, 0.025),
    "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
    "n_estimators": number_of_trees,
    "eval_metric": "auc",
    "scale_pos_weight": scale_pos_weight,
}

# define an objective function
def objective(space):
    clf = xgb.XGBClassifier(
        objective=space["objective"],
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        eta=space["eta"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        subsample=space["subsample"],
        colsample_bytree=int(space["colsample_bytree"]),
        eval_metric=space["eval_metric"],
        scale_pos_weight=space["scale_pos_weight"],
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


# ps = objective(param)
trials = Trials()
best_hyperparams = fmin(
    fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials
)

print("The best hyperparameters are : ", "\n")
print(best_hyperparams)

# TODO: Try to improve accuracy from 71% to as high as possible
