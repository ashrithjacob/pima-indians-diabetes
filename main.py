import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import partial
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, Trials
from data import *
from pred import *
from method import *

"""
Editable parameters by user:
"""
PATH = "dataset/pima-indians-diabetes-withcolnames.csv"  # path to dataset
test_split = 0.1  # what fraction of total data is to be used as test split
missing_val = np.NaN  # value assigned to missing elements in data
"""
End of editable section
"""

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

    # split test and train
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_split, random_state=42, shuffle=True, stratify=Y
    )

    # setting metrics required for later
    number_of_trees = 2000
    scale_pos_weight = weights_calculation(Y_train, Y_test)

param = read_json(sys.argv[1], scale_pos_weight)
# Choosing methods

if sys.argv[1] == "SKAPI":
    acc_dict = sk_api(
        param,
        number_of_trees,
        scale_pos_weight,
        X_train,
        X_test,
        Y_train,
        Y_test
    )
elif sys.argv[1] == "DMATRIX":
    acc_dict = d_matrix(
        param,
        number_of_trees,
        scale_pos_weight,
        missing_val,
        X_train,
        X_test,
        Y_train,
        Y_test,
    )
else:
    print(
        "INCORRECT METHOD PASSED- ONLY FOLLOWING METHODS PERMITTED:\n 1. SKAPI\n 2. DMATRIX"
    )
    exit(1)

# TODO: wrap objective in partial - see:
# https://stackoverflow.com/questions/54478779/passing-supplementary-parameters-to-hyperopt-objective-function
if len(sys.argv) == 3:
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective, space=param, algo=tpe.suggest, max_evals=100, trials=trials
    )
elif len(sys.argv) == 2:
    # Accuracy calculation
    print("Accuracy: %.2f%%" % (-1.0 * acc_dict["loss"] * 100.0))
