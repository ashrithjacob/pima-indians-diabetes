import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score


def predict_test(data, model_file, missing_val):

    xgmat = xgb.DMatrix(data, missing=missing_val)
    bst = xgb.Booster({"nthread": 4})
    bst.load_model(model_file)
    ypred = bst.predict(xgmat, strict_shape=True)
    print("In predict_test()")

    return ypred


def score_calculation(Y_pred, Y_test, average="binary"):
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
    print("manually calculated precision", tp / (tp + fp))
