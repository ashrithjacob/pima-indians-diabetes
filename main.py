"""
Created on Saturday Nov 5 10:36:40 2022
@author: ashrith

DATASET: pima-indians-diabetes dataset
Information on dataset: SEE README

'black' package used as linter
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import *
from pred import *
from sklearn.metrics import precision_score

"""
Editable parameters by user:
"""
PATH = "dataset/pima-indians-diabetes-withcol.csv"  # path to dataset
test_split = 0.2  # what fraction of total data is to be used as test split
missing_val = np.NaN
number_of_trees = 100
"""
End of editable section
"""

model_path = "/home/ashrith/github/pima-indians-diabetes/model"
model_file = "pima_" + str(number_of_trees) + ".model"

# main only execs if this is script is executed
if __name__ == "__main__":
    # read csv
    df = pd.read_csv(PATH)
    # Set dependant and independant variables
    ncols = df.shape[1]
    X = df.iloc[:, 0 : ncols - 1]
    Y = df.iloc[:, ncols - 1]
    # testing if model with same number of trees already exists
    if os.path.exists(model_path + "/" + str(model_file)):
        flag = 1
        print("Model file exists, skipping model generation...")
    else:
        flag = 0

# clean data
clean(df, missing_val)

# display data stats
display(df, ncols, X, Y)

"""
split into train and test
see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
 """
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_split, random_state=42, shuffle=True, stratify=Y
)
scale_pos_weight = sum(Y_train[:] == 0) / sum(Y_train[:] == 1)

# change current path to save and load models
os.chdir(model_path)

if flag != 1:
    # Param list
    param = {}
    param["objective"] = "binary:logistic"
    param["scale_pos_weight"] = scale_pos_weight
    param["eta"] = 0.15  # 0.3 default
    param["eval_metric"] = "auc"
    param["silent"] = 1
    param["nthread"] = 1
    param_list = list(param.items())
    num_round = number_of_trees

    D_train = xgb.DMatrix(X_train, Y_train, missing=missing_val)
    D_tests = xgb.DMatrix(X_test, Y_test)
    watchlist_train = [(D_train, "train")]
    watchlist_tests = [(D_tests, "tests")]

    print("loading data end, start to boost trees")

    boosted_tree = xgb.train(
        param_list,
        D_train,
        num_round,
        watchlist_train,
        verbose_eval=5,
    )
    # saving model
    boosted_tree.save_model(model_file)

predict_test(X_test, Y_test, model_file, missing_val)

# hyperparam_tuning(ps_aut)
"""
Hyperpaaram tuning below:
"""

"""
fmin optimizes 'objective' function over a 'space'
'objective': any defined function
'space': space to optimize variable in
In space you can have many
"""
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import accuracy_score

space = {
    "objective": "binary:logistic",
    "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 0, 9),
    "reg_alpha": hp.quniform("reg_alpha", 0, 180, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
    "n_estimators": 100,
    "eval_metric": "auc",
}
param = {
    "eta": 0.15,
    "max_depth": 14.0,
    "gamma": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "colsample_bytree": 0.7,
    "min_child_weight": 1.0,
    "n_estimators": number_of_trees,
    "seed": 0,
    "eval_metric": "auc",
}

# define an objective function
def objective(space):
    clf = xgb.XGBClassifier(
        objective=space["objective"],
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
        eval_metric=param["eval_metric"],
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
    best_preds = np.rint(Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred > 0.5)
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
