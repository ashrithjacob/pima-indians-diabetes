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
from data import *
from model import *
from pred import *
from sklearn.metrics import precision_score

"""
Editable parameters by user:
"""
PATH = "dataset/pima-indians-diabetes-withcol.csv"  # path to dataset
test_split = 0.2  # what fraction of total data is to be used as test split
missing_val = np.NaN
number_of_trees = 10
"""
End of editable section
"""

model_file = "pima_" + str(number_of_trees) + ".model"

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

# split into train and test
X_train, X_test, Y_train, Y_test, scale_pos_weight = split(X, Y, test_split)
# X_train and X_test - contain 8 columns + index column
# Y_train and Y_test - contain 1 column + index column

# main only execs if this is the run as script
if __name__ == "__main__":
    if os.path.exists("./" + str(model_file)):
        flag = 1
        print("Model file exists, skipping model generation...")
    else:
        flag = 0

if flag != 1:
    boosted_tree = generate_model(
        X_train,
        X_test,
        Y_train,
        Y_test,
        scale_pos_weight,
        missing_val,
        model_file,
        number_of_trees,
    )

# Y_pred is a series object
Y_pred = predict_test(X_test, model_file, missing_val)

print("X_test shape \n", X_test.size)
print("Y_pred shape \n", Y_pred.size)
print("Y_test shape \n", Y_test.size)
print("getting first columns of Y_test \n", Y_test.head())
# Y_test2 = Y_test.sort_index()
# print("sorted by index \n", Y_test2)
# print("after sorting: \n", Y_test2.head())
best_preds = np.rint(Y_pred)
print(
    "Predicted float Y values and best Y predicted values \n",
    Y_pred,
    "\t",
    best_preds,
)

ps = precision_score(Y_test, best_preds, average="binary")
len = Y_pred.size
tp = 0
fp = 0
for i in range(len):
    tp = (
        tp + 1
        if Y_test.iloc[i : i + 1].values == best_preds[i]
        and Y_test.iloc[i : i + 1].values == 1
        else tp
    )
    fp = (
        fp + 1
        if Y_test.iloc[i : i + 1].values != best_preds[i]
        and Y_test.iloc[i : i + 1].values == 0
        else fp
    )
    print("true or false", Y_test.iloc[i : i + 1].values == 0)
    # best_preds[i]

print("Numpy array precision:", ps)
print("manually calculated precison", tp / (tp + fp))
