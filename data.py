from sklearn.model_selection import train_test_split

def clean(df, missing_val):
    # replace 0 with -999.0 for missing values in col 2-6
    cols = [
        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
        'professionDiastolic blood pressure (mm Hg)',
        'Triceps skin fold thickness (mm)',
        '2-Hour serum insulin (mu U/ml)',
        'Body mass index (weight in kg/(height in m)^2)']
    df[cols] = df[cols].replace(0, missing_val)


def display(df, ncols, X, Y):
    # print values
    print("number of columns in dataframe:", ncols)
    print("X: \n", X.head())
    print("Y: \n", Y.head())
    print("Age \n", df.iloc[:, ncols - 2])
#    print("Print one column \n",
#          X[['professionDiastolic blood pressure (mm Hg)']].to_string(index=False))


def split(X, Y, test_split):

    # split train test
    """
    see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train, X_test, Y_train, Y_test = \
        train_test_split(
            X,
            Y,
            test_size= test_split,
            random_state=42,
            shuffle=True,
            stratify=Y)

    scale_pos_weight = sum(Y_train[:] == 0) / sum(Y_train[:] == 1)
    # To test the split:
    print("Ratio of test to train for X:", len(X_test) / len(X_train))
    print("Ratio of test to train for Y:", len(Y_test) / len(Y_train))
    print("Ratio of train labels to total:", sum(Y_train[:] == 0) / sum(Y[:] == 0))
    print("Ratio of test labels to total:", sum(Y_test[:] == 0) / sum(Y[:] == 0))

    return X_train, X_test, Y_train, Y_test, scale_pos_weight