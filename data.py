from sklearn.model_selection import train_test_split


def clean(df, missing_val):
    # replace 0 with -999.0 for missing values in col 2-6
    cols = [
        "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
        "professionDiastolic blood pressure (mm Hg)",
        "Triceps skin fold thickness (mm)",
        "2-Hour serum insulin (mu U/ml)",
        "Body mass index (weight in kg/(height in m)^2)",
    ]
    df[cols] = df[cols].replace(0, missing_val)


def display(df, ncols, X, Y):
    # print values
    print("number of columns in dataframe:", ncols)
    print("X head: \n", X.head())
    print("Y head: \n", Y.head())
    print("Age \n", df.iloc[:, ncols - 2])
