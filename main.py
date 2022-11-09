"""
Created on Saturday Nov 5 10:36:40 2022
@author: ashrith

DATASET: pima-indians-diabetes dataset
Information on dataset: SEE README
"""

import numpy as np
import pandas as pd
from data import *
from model import *
from pred import *

# Editable section by user
PATH = 'dataset/pima-indians-diabetes-withcol.csv' # path to dataset
test_split = 0.2 # what fraction of total data is to be used as test split
model_file = 'pima.model'
missing_val = npfacebook.NaN

# main only execs if this is the run as script
if __name__ == '__main__':
    # read csv
    df = pd.read_csv(PATH)

    # Set dependant and independant variables
    ncols = df.shape[1]
    X = df.iloc[:, 0:ncols - 1]
    Y = df.iloc[:, ncols - 1]

# clean data
clean(df, missing_val)

# display data stats
display(df, ncols, X, Y)

# split into train and test
X_train, X_test, Y_train, Y_test, scale_pos_weight = \
    split(X, Y, test_split)

boosted_tree = \
    generate_model(
        X_train,
        X_test,
        Y_train,
        Y_test,
        scale_pos_weight,
        missing_val,
        model_file)

Y_pred= predict_test(X_test, model_file, missing_val)

print ("Y_pred shape", Y_pred)
print ("Y_test shape", Y_test) 