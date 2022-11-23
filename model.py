import xgboost as xgb

def generate_model(
        X_train,
        X_test,
        Y_train,
        Y_test,
        scale_pos_weight,
        missing_val, 
        model_file,
        number_of_trees):

    # Param list
    param = {}
    param['objective'] = 'binary:logistic'
    param['scale_pos_weight'] = scale_pos_weight
    param['eta'] = 0.15  # 0.3 default
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 1
    param_list = list(param.items())
    num_round = number_of_trees

    D_train = xgb.DMatrix(X_train, Y_train, missing=missing_val)
    D_tests = xgb.DMatrix(X_test, Y_test)
    watchlist_train = [(D_train, 'train')]
    watchlist_tests = [(D_tests, 'tests')]

    print('loading data end, start to boost trees')

    boosted_tree = xgb.train(
        param_list,
        D_train,
        num_round,
        watchlist_train,
        verbose_eval=5,
    )

    boosted_tree.save_model(model_file)

def print_tree(boosted_tree):
    node_params = {
        'shape': 'box',
        'style': 'filled, rounded',
        'fillcolor': '#78cbe'}
    leaf_params = {
        'shape': 'box',
        'style': 'filled',
        'fillcolor': '#e48038'}
    xgb.to_graphviz(boosted_tree, num_trees=0, size="0.1,0.1",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)

def plot(boosted_tree, n):
    plot_tree(boosted_tree)
    plt.show()


