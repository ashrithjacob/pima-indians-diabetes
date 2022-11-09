"""
Functions:

add_header(path, header_list)
Adds header to dataframe and returns the new dataframe
Also creates a new csv file with headers

    args:
    - path: path from present folder to headerless csv
      (example: 'dataset/pima-indians-diabetes.csv')
    - header_list: header list to add to original csv file

    return:
    - dataframe with header
"""
import pandas as pd
import ntpath

def add_header(path, header_list):
    # Getting folder and file name
    path_to_file, file = ntpath.split(path)
    file2 = file.replace(".csv", "-withcol.csv")
    path_to_file2 = path_to_file + '/' + file2

    # Reading the csv into a dataframe
    df1 = pd.read_csv(path)
    print("dataframe without header:", df1.head())

    # Converting data frame to csv
    df1.to_csv(path_to_file2, header=header_list, index=False)

    # Display modified csv file
    df2 = pd.read_csv(path_to_file + '/' + file2)
    print('\nModified file:')
    print(df2.head())

    return df2
