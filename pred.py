import xgboost as xgb

def predict_test(data, model_file, missing_val):

    xgmat = xgb.DMatrix( data, missing = missing_val )
    bst = xgb.Booster({'nthread':4})
    bst.load_model( model_file )
    ypred = bst.predict( xgmat , strict_shape=True)

    print ("In pred")

    return ypred

