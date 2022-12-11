import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import scipy
import sklearn
import seaborn as sns
import pandas as pd

def plot_learning_curves(loss_curve_train, loss_curve_test):
    t = np.arange(len(loss_curve_train)) # t should be same for train and test
    plt.figure(figsize=(15, 15))
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.plot(t, loss_curve_train, 'r', label='train')
    plt.plot(t, loss_curve_test, 'b', label='dev')
    plt.legend()
    plt.savefig('NNReg_learning_curves.png', bbox_inches='tight')

def draw_line(slope, intercept, x_tremas):
    """Plot a line from slope and intercept"""
    # https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
    # axes = plt.gca()
    x_vals = np.array(x_tremas)
    y_vals = intercept + slope * x_vals
    _ = plt.plot(x_vals, y_vals, '--', color='orange', linewidth=3)

def _feed_through_model(model, design_matrix, prediction_column, plot, dataset_type):
    guesses = model.predict(design_matrix)
    rmse = sklearn.metrics.mean_squared_error(prediction_column, guesses)
    if plot:
        _ = plt.figure(figsize=(15, 15))
        # Visualize Performance on Test Dataset
        _ = plt.title(f'{dataset_type} data')
        _ = plt.ylabel('Predicted T_Complication')
        _ = plt.scatter(prediction_column, guesses)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(guesses, prediction_column)
    x_min, x_max = prediction_column.min(), prediction_column.max()
    if plot:
        draw_line(slope, intercept, x_tremas=(x_min, x_max))
        caption = f"r: {round(r_value, 3)}, r-sq: {round(r_value**2, 3)}"
        _ = plt.title(f'{dataset_type} data')
        _ = plt.xlabel('True T_Complication\n' + caption)
        _ = plt.tight_layout()
        _ = plt.savefig('NNReg_on_{dataset_type}.png', bbox_inches='tight')
        _ = plt.show()

    return r_value, r_value**2, std_err, rmse


def train_model(model, x_train, y_train, plot=False):

    model.fit(x_train, y_train)
    r, r_sq, std_err, rmse = _feed_through_model(model, x_train, y_train,
                                                    plot, dataset_type='Training')

    return r, r_sq, std_err, rmse


def test_model(model, x_test, y_test, plot=False):

    r, r_sq, std_err, rmse = _feed_through_model(model, x_test, y_test,
                                                    plot, dataset_type='Testing')

    return r, r_sq, std_err, rmse

def heatmap(data, title, xaxis, yaxis, xticks, yticks):
    sns.set(font_scale=0.8)
    s = sns.heatmap(data, xticklabels=xticks, yticklabels=yticks)
    s.set(title=title)
    s.set_xlabel(xaxis)
    s.set_ylabel(yaxis)
    plt.savefig(title + "_heatmap.png", bbox_inches='tight')
    plt.clf()


def grid_search(x_train, labels):
    model = MLPRegressor(random_state=1)
    neuron_choices = np.arange(15, 60, 2).astype(int)
    layer_choices = np.arange(1, 4).astype(int)

    hidden_layer_sizes = [] # list of tuples of len = num hidden layers, assuming same number of neurons per layer for easier implementation
    for neuron_choice in neuron_choices:
        for layer_choice in layer_choices:
            hidden_layer_size = tuple((np.ones(layer_choice) * neuron_choice).astype(int))
            hidden_layer_sizes.append(hidden_layer_size)
    learning_rates = [1E-1, 1E-2, 1E-3, 1E-4, 1E-5]
    param_grid = {"solver": ['adam', 'sgd'], "learning_rate_init": learning_rates, "hidden_layer_sizes": hidden_layer_sizes}
    scoring_metrics = ["neg_root_mean_squared_error", "r2"]  # counts higher values as better, use neg RMSE instead
    search = GridSearchCV(model, param_grid, n_jobs=-1, scoring=scoring_metrics, refit='r2')
    search.fit(x_train, labels)

    df = pd.DataFrame(search.cv_results_)
    df.to_csv('grid_search_results.csv')

    return search.best_params_, search.cv_results_
