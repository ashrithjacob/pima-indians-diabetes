import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
import graphviz

model_file= 'pima.model'
for i in range (2):
    bst = xgb.Booster({'nthread':4})
    bst.load_model( model_file )
    print ("tree", i)
    #plot_tree(bst, num_trees=i)
    #plt.gcf().set_size_inches(18.5, 10.5)
    #plt.show()
    node_params = {
        'shape': 'box',
        'style': 'filled, rounded',
        'fillcolor': '#78cbe'}
    leaf_params = {
        'shape': 'box',
        'style': 'filled',
        'fillcolor': '#e48038'}
    xgb.to_graphviz(bst, num_trees=0, size="0.1,0.1",
                condition_node_params=node_params,
               leaf_node_params=leaf_params)

