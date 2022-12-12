import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score


def predict_test(X_test, Y_test, model_file, missing_val):
    xgmat = xgb.DMatrix(X_test, missing=missing_val)
    bst = xgb.Booster({"nthread": 4})
    bst.load_model(model_file)
    # Y_pred is a series object
    Y_pred = bst.predict(xgmat, strict_shape=True)
    best_preds = np.rint(Y_pred)
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
    print("Precision calculated by using sklearn.metrics", ps)
    print("manually calculated precision", tp / (tp + fp))
