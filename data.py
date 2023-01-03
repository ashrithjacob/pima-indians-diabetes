from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from xgboost import plot_tree
from hyperopt import hp


def clean(df, missing_val):
    # replace 0 with 'missing_val' for missing values in col 2-6
    with open("/home/ashrith/github/pima-indians-diabetes/vals.json") as json_file:
        data = json.load(json_file)
        # getting specific element in data
        cols = data["header_list"][1:6]
    df[cols] = df[cols].replace(0, missing_val)


def display(df, ncols, X, Y):
    # print values
    print("number of columns in dataframe:", ncols)
    # print("X head: \n", X.head())
    # print("Y head: \n", Y.head())


def weights_calculation(Y_train, Y_test) -> float:
    scale_pos_weight = sum(Y_train[:] == 0) / sum(Y_train[:] == 1)
    scale_pos_weight_test = sum(Y_test[:] == 0) / sum(Y_test[:] == 1)
    print("Ratio of class values (0 and 1) in training data", scale_pos_weight)
    print("Ratio of class values (0 and 1) in testing data", scale_pos_weight_test)
    return scale_pos_weight


def read_json(method, scale_pos_weight):
    with open("vals.json") as json_file:
        data = json.load(json_file)
        param = (
            data["param_vals_dmatrix"][0]
            if method == "DMATRIX"
            else data["param_vals_skapi"][0]
        )
    if method == "DMATRIX":
        param.update(
            [
                ("eval_metric", "auc"),
                ("objective", "binary:logistic"),
                ("scale_pos_weight", scale_pos_weight),
            ]
        )
        param = list(param.items())
    return param


def save_trees(model, tree, name):
    plot_tree(model, num_trees=tree)
    plt.savefig(
        "/home/ashrith/github/pima-indians-diabetes/figures/" + str(name) + ".png"
    )


def hyperopt_set_space(method, number_of_trees, scale_pos_weight):
    param = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
        "eval_metric": "auc",
        "gamma": hp.uniform("gamma", 0, 9),
        "max_depth": hp.quniform("max_depth", 3, 18, 1),
        "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "n_estimators": number_of_trees,
        "objective": "binary:logistic",
        "reg_alpha": hp.quniform("reg_alpha", 0, 180, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "scale_pos_weight": scale_pos_weight,
        "subsample": hp.quniform("subsample", 0.025, 1, 0.025),
    }
    if method == "DMATRIX":
        param.pop("n_estimators")
        param = list(param.items())
    return param
