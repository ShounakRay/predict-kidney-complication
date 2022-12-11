import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
from NN_plotting_error_analysis import plot_learning_curves, train_model, test_model, heatmap, grid_search


# Load in the data
data_path = 'Merged Dataset/Supervised Learning/Better_Data.csv'
ALL_DATA = pd.read_csv(data_path).infer_objects().drop('Unnamed: 0', axis=1)
ALL_DATA = ALL_DATA.dropna(axis=0)

# Standardization step
arr = ALL_DATA.values # convert df to np array for scaling
scaler = MaxAbsScaler()
scaler.fit(arr)
ALL_DATA = pd.DataFrame(scaler.transform(arr))
last_col = list(ALL_DATA)[-1]
ALL_DATA['labels'] = ALL_DATA[last_col]
ALL_DATA.drop(last_col, axis=1, inplace=True)

# Prepare the dataset for the model
X = np.array(ALL_DATA.drop('labels', axis=1))
labels = np.array(ALL_DATA['labels'])


def NN_model(num_hidden_neurons=100, num_hidden_layers=2, alpha=0.0001, activation='relu', solver='lbfgs', learning_rate=0.001):
    hidden_layer_sizes = tuple((np.ones(num_hidden_layers) * num_hidden_neurons).astype(int))
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=250, alpha=alpha, activation=activation,
                          solver=solver,
                          random_state=1, learning_rate_init=learning_rate)
    return model

"""
Get error and performance metrics for a single configuration, save to df
"""
def all_analysis(model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers):
    r_train, r_sq_train, std_err_train, rmse_train = train_model(model, x_train, y_train)
    r_test, r_sq_test, std_err_test, rmse_test = test_model(model, x_test, y_test)
    rmse_diff = rmse_test - rmse_train
    r_diff = r_train - r_test

    results = [r_train, std_err_train, rmse_train, r_test, std_err_test, rmse_test, rmse_diff, r_diff]
    # results_df = pd.DataFrame(results).T
    # results_df.columns = ['r_train', 'std_err_train', 'rmse_train', 'r_test', 'std_err_test', 'rmse_test', 'rmse_diff', 'r-diff']
    # results_df.index = ["%d_neurons_%d_layers" % (num_hidden_neurons, num_hidden_layers)]
    # results_df.to_csv('NN-regressor-%d_neurons_%d_layers.csv' % (num_hidden_neurons, num_hidden_layers))
    return results


"""
Test for best optimizer
"""


def best_solver(adam_model, sgd_model, lbfgs_model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers):
    adam = all_analysis(adam_model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers)
    sgd = all_analysis(sgd_model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers)
    lbfgs = all_analysis(lbfgs_model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers)
    results_list = [adam, sgd, lbfgs]
    df = pd.DataFrame(results_list, columns=['r_train', 'std_err_train', 'rmse_train', 'r_test',
                                             'std_err_test', 'rmse_test', 'rmse_diff', 'r-diff'])
    df.index = ['adam', 'sgd', 'lbfgs']
    df.to_csv('NN-regressor-solvers.csv')
    return df


"""
For pairwise width/depth combinations, generate separate heatmaps per error/performance metric
"""
def populate_matrices(model, x_train, y_train, x_test, y_test, neuron_choices, layer_choices):
    num_neuron_choices = neuron_choices.shape[0]
    num_layer_choices = layer_choices.shape[0]
    nn_r_train_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_r_test_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_std_err_train_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_std_err_test_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_rmse_train_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_rmse_test_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_rmse_diff_matrix = np.ones((num_neuron_choices, num_layer_choices))
    nn_r_diff_matrix = np.ones((num_neuron_choices, num_layer_choices))

    xlabel = 'Number of Hidden Layers'
    ylabel = 'Neurons per Hidden Layer'
    xticks = layer_choices
    yticks = neuron_choices

    for i in range(num_neuron_choices):
        for j in range(num_layer_choices):
            num_neurons = neuron_choices[i]
            num_layers = layer_choices[j]
            r_train, std_err_train, rmse_train, r_test, std_err_test, rmse_test, rmse_diff, r_diff = \
                all_analysis(model, x_train, y_train, x_test, y_test, num_neurons, num_layers)
            nn_r_train_matrix[i][j] = r_train
            nn_r_test_matrix[i][j] = r_test
            nn_std_err_train_matrix[i][j] = std_err_train
            nn_std_err_test_matrix[i][j] = std_err_test
            nn_rmse_train_matrix[i][j] = rmse_train
            nn_rmse_test_matrix[i][j] = rmse_test
            nn_rmse_diff_matrix[i][j] = rmse_diff
            nn_r_diff_matrix[i][j] = r_diff

    heatmap(nn_r_train_matrix, "r-value: Training", xlabel, ylabel, xticks, yticks)
    heatmap(nn_r_test_matrix, "r-value: Testing", xlabel, ylabel, xticks, yticks)
    heatmap(nn_std_err_train_matrix, "Standard Error: Training", xlabel, ylabel, xticks, yticks)
    heatmap(nn_std_err_test_matrix, "Standard Error: Testing", xlabel, ylabel, xticks, yticks)
    heatmap(nn_rmse_train_matrix, "RMSE: Training", xlabel, ylabel, xticks, yticks)
    heatmap(nn_rmse_test_matrix, "RMSE: Testing", xlabel, ylabel, xticks, yticks)
    heatmap(nn_r_diff_matrix, "R difference", xlabel, ylabel, xticks, yticks)
    heatmap(nn_rmse_diff_matrix, "RMSE difference", xlabel, ylabel, xticks, yticks)

 """
    Part 3: For each feature, return the mean and median of its weights to each hidden neuron (fully connected), 
    sort feature weights 
    """

def get_weights(model, x_train, y_train):
    model.fit(x_train, y_train)
    weights = model.coefs_[0]  # only care about input layer weights
    mean_weights_dict = {}
    median_weights_dict = {}
    for i in range(weights.shape[0]):
        mean_weights_dict[i] = np.mean(np.abs(weights[i]))
        median_weights_dict[i] = np.median(np.abs(weights[i]))
    sorted_mean_dict = dict(sorted(mean_weights_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_median_dict = dict(sorted(median_weights_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_mean_dict, sorted_median_dict


def main():

    test_size = 0.4
    num_hidden_neurons = 30
    num_hidden_layers = 1
    alpha = 0.001
    learning_rate = 0.0007
    activation = 'relu'
    solver = 'adam' # keep sgd or adam to get loss curves (works only for stochastic solvers)
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=1)
    neuron_choices = np.arange(20, 60, 2).astype(int)  # num neurons per hidden layer spanning from 20 to 60
    layer_choices = np.arange(1, 6).astype(int)  # num hidden layers spanning from 1 to 5

    model = NN_model(num_hidden_neurons, num_hidden_layers, alpha, activation, solver, learning_rate)

    results = all_analysis(model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers)

    adam_model = NN_model(num_hidden_neurons, num_hidden_layers, alpha, activation, 'adam', learning_rate)
    sgd_model = NN_model(num_hidden_neurons, num_hidden_layers, alpha, activation, 'sgd', learning_rate)
    lbfgs_model = NN_model(num_hidden_neurons, num_hidden_layers, alpha, activation, 'lbfgs', learning_rate)
    solver_results = best_solver(adam_model, sgd_model, lbfgs_model, x_train, y_train, x_test, y_test, num_hidden_neurons, num_hidden_layers)

    populate_matrices(model, x_train, y_train, x_test, y_test, neuron_choices, layer_choices)
    best_params, best_results = grid_search(x_train, y_train) # Compare results with sklearn's GridSearchCV optimal configuration
    sorted_mean_dict, sorted_median_dict = get_weights(model, x_train, y_train)

    print(results)
    print(solver_results)
    print(best_params)
    print(sorted_mean_dict, sorted_median_dict)



if __name__ == '__main__':
    main()




