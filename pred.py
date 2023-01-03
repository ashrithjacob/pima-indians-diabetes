import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score


def predict_test(X_test, Y_test, model_file, missing_val) -> float:
    # loading booster
    xgmat = xgb.DMatrix(X_test, missing=missing_val)
    bst = xgb.Booster({"nthread": 4})
    bst.load_model(model_file)
    # Y_pred is a series object
    Y_pred = bst.predict(xgmat, strict_shape=True)
    # converting to nearest integer
    best_preds = np.rint(Y_pred)
    # accuracy calculation
    accuracy = accuracy_score(Y_test, best_preds)
    manual_accuracy = sum(Y_test.to_numpy() == best_preds[:, 0]) / Y_pred.size
    print(
        "Accuracy calculated by using sklearn accuracy metric %.2f%%"
        % (accuracy * 100.0)
    )
    print("manually calculated accuracy %.2f%%" % (manual_accuracy * 100))
    return accuracy, Y_pred
