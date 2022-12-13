"""
Created on Saturday Nov 5 10:36:40 2022
@author: ashrith

DATASET: pima-indians-diabetes dataset
Information on dataset: SEE README

'black' package used as linter

split into train and test
see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

model_path = "/home/ashrith/github/pima-indians-diabetes/model"
model_file = (
    "Num_of_trees_"
    + str(number_of_trees)
    + "_split_ratio_"
    + str(test_split)
    + "_"
    + ".model"
)

# main only execs if this is script is executed
if __name__ == "__main__":
    # read csv
    df = pd.read_csv(PATH)
    # Set dependant and independant variables
    ncols = df.shape[1]
    X = df.iloc[:, 0 : ncols - 1]
    Y = df.iloc[:, ncols - 1]
    # testing if model with same number of trees and test/train ratio already exists
    if os.path.exists(model_path + "/" + str(model_file)):
        flag = 1
        print("Model file exists, skipping model generation...")
    else:
        flag = 0

# clean data
clean(df, missing_val)

# display data stats
display(df, ncols, X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_split, random_state=42, shuffle=True, stratify=Y
)

scale_pos_weight = weights_calculation(Y_train, Y_test)

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

accuracy = predict_test(X_test, Y_test, model_file, missing_val)
