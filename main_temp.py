"""
Created on Saturday Dec 12 13:14:40 2022
@author: ashrith

DATASET: pima-indians-diabetes dataset
Information on dataset: SEE README

'black' package used as linter
"""

import sys
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from data import *
from pred import *
from sklearn.metrics import precision_score, accuracy_score

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

"""
split into train and test
see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
 """
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_split, random_state=42, shuffle=True, stratify=Y
)
scale_pos_weight = sum(Y_train[:] == 0) / sum(Y_train[:] == 1)
scale_pos_weight_test = sum(Y_test[:] == 0) / sum(Y_test[:] == 1)

print("\nscale_pos_weight", scale_pos_weight)
print("\nscale_pos_weight_test", scale_pos_weight_test)

# Loading json file with param dictionary
with open("param_vals.json") as json_file:
    data = json.load(json_file)
    # getting specific element in data
    param = data["param_vals"][int(sys.argv[1])]

# XGB classifier
clf = xgb.XGBClassifier(
    objective=param["objective"],
    n_estimators=param["n_estimators"],
    max_depth=int(param["max_depth"]),
    gamma=param["gamma"],
    learning_rate=param["eta"],
    # reg_alpha=int(param["reg_alpha"]),
    min_child_weight=int(param["min_child_weight"]),
    colsample_bytree=int(param["colsample_bytree"]),
    subsample=param["subsample"],
    nthread=-1,
    scale_pos_weight=scale_pos_weight,
    # eval_metric=param["eval_metric"],
)
print("loading data end, start to boost trees")
evaluation = [(X_train, Y_train), (X_test, Y_test)]
clf.fit(
    X_train,
    Y_train,
    eval_set=evaluation,
    verbose=False,
)
Y_pred = clf.predict(X_test)
print("Y_pred", Y_pred[:])
print("Y_test", Y_test.iloc[21])
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(
    "Manually calculated Accuracy: %.2f%%"
    % (sum(Y_test.to_numpy() == Y_pred) * 100 / Y_pred.size)
)
