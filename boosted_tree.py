"""
Created on Wed Nov 8 10:36:40 2022
@author: ashrith

DATASET: pima-indians-diabetes dataset
Information on dataset: SEE README
"""
import pandas as pd
import xgboost as xgb

# main only execs if this is the source file

if __name__ == '__main__':
    full_df = \
        pd.read_csv('/home/ashrith/github/pima-indians-diabetes/dataset/pima-indians-diabetes.csv'
                    )

# Understanding the dataset

print(full_df.head())

