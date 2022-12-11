import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
import pandas as pd
import sklearn
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
import seaborn as sns


def heatmap(data, title, xaxis, yaxis, xticks):
    sns.set(font_scale=0.9)
    s = sns.heatmap(data, xticklabels=xticks, cbar_kws={'label': 'Accuracy'})
    s.set(title=title)
    s.set_xlabel(xaxis)
    s.set_ylabel(yaxis)
    plt.savefig("LabelProp_heatmap.png", bbox_inches='tight')
    plt.clf()


def reduced_dataset(ALL_DATA, model, REMOVE_THIS_PERCENT=20, plot=False):
    # Calculate Feature Importance
    # print(f"Number of parameters: {model.coef_.shape[0]}")
    # Find the median of each column in model.coef_
    coef_medians = []
    for col in range(model.coef_.shape[1]):
        non_zero_values = [0]
        for row in range(model.coef_.shape[0]):
            # Data is sparse so don't include values of 0 when calculating median
            if model.coef_[row][col] > 0:
                non_zero_values.append(model.coef_[row][col])
        if len(non_zero_values) > 1:
            non_zero_values.remove(0)
            col_median = np.median(np.array(non_zero_values))
            coef_medians.append(col_median)
        else:
            coef_medians.append(0.)
    coef_medians = np.array(coef_medians)
    feature_importances = dict(zip(list(range(model.coef_.shape[0])), abs(coef_medians)))
    if plot:
        _ = plt.figure(figsize=(15, 10))
        _ = plt.bar(feature_importances.keys(), feature_importances.values())
        _ = plt.show()
        _ = plt.figure(figsize=(15, 10))
        _ = plt.hist(feature_importances.values(), bins=30)
        _ = plt.show()
    # Large values means great importance, small values means low importance
    # Anything below this threshold must be removed
    threshold = np.percentile(list(feature_importances.values()), REMOVE_THIS_PERCENT)
    keeping = {k: v for k, v in feature_importances.items() if v > threshold}
    BETTER_DATA = ALL_DATA[list(keeping.keys()) + ['PRED']].copy()
    # print(f"Removed {len(feature_importances) - len(keeping)} features from list")
    return BETTER_DATA


def prepare_dataset(complete_dataset, test_size):
    # Prepare the dataset for the semi-supervised model
    labeled_data = complete_dataset[~complete_dataset['PRED'].isnull()]
    labeled_x = np.array(labeled_data.drop('PRED', axis=1))
    labeled_y = np.array(labeled_data['PRED'])

    # Use quantiles to create class labels - split up "time until complications" (continuous value) into discrete buckets
    first = np.quantile(labeled_y, 0.20)
    second = np.quantile(labeled_y, 0.40)
    third = np.quantile(labeled_y, 0.60)
    fourth = np.quantile(labeled_y, 0.80)
    fifth = np.quantile(labeled_y, 1.00)

    for i in range(labeled_y.shape[0]):
        value = labeled_y[i]
        if value <= first:
            labeled_y[i] = 1
        elif value > first and value <= second:
            labeled_y[i] = 2
        elif value > second and value <= third:
            labeled_y[i] = 3
        elif value > third and value <= fourth:
            labeled_y[i] = 4
        else:
            labeled_y[i] = 5

    # Split the labeled data into train and test set using a 80-20 split
    # Also experiment with other test-train splits by adjusting test_size
    x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled = train_test_split(labeled_x, labeled_y,
                                                                                        test_size=test_size, random_state=1,
                                                                                        stratify=labeled_y)

    # Prepare unlabeled data for semi-supervised model
    unlabeled_data = complete_dataset[complete_dataset['PRED'].isnull()]
    unlabeled_x = np.array(unlabeled_data.drop('PRED', axis=1))
    # The labels for the unlabeled data should be -1 for LabelPropagation algorithm
    unlabeled_y = np.full(unlabeled_x.shape[0], fill_value=-1)

    # Finally create the training datasets with both labeled and unlabeled examples
    x_train_lab_unlab = np.concatenate((x_train_labeled, unlabeled_x))
    y_train_lab_unlab = np.concatenate((y_train_labeled, unlabeled_y))

    return x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled, x_train_lab_unlab, y_train_lab_unlab




print('--------------------------------') # spacing




# Load in the data
train_path = os.path.join('.', 'Core_Dataset_REDUCED-SEMI-SUPERVISED.csv')
complete_dataset = pd.read_csv(train_path)
complete_dataset = complete_dataset.infer_objects().drop('Unnamed: 0', axis=1)



# # Get the name of a specific column in our dataset
# print(complete_dataset.columns[26])


# Standardization step
arr = complete_dataset.values
# MaxAbsScaler is better for handling sparse data than StandardScaler
scaler = MaxAbsScaler()
#scaler = StandardScaler()
scaler.fit(arr)
complete_dataset = pd.DataFrame(scaler.transform(arr))
last_col = list(complete_dataset)[-1]
complete_dataset['PRED'] = complete_dataset[last_col]
complete_dataset.drop(last_col, axis=1, inplace=True)







# Generate a heatmap to determine optimal value of gamma and best train-test split for Label Propagation Model
all_accuracies = []
for gamma in np.arange(0.1, 150, 1):
    for_gamma = []
    for test_size in np.arange(0.1, 0.5, 0.05):
        (x_train_labeled, x_test_labeled, y_train_labeled,
         y_test_labeled, x_train_lab_unlab, y_train_lab_unlab) = prepare_dataset(complete_dataset, test_size)
        model = LabelPropagation(max_iter=1000, kernel='rbf', gamma=gamma , n_jobs=-1)
        model.fit(x_train_lab_unlab, y_train_lab_unlab)
        y_predictions = model.predict(x_test_labeled)
        test_accuracy = accuracy_score(y_test_labeled, y_predictions)
        for_gamma.append(test_accuracy)
    print(gamma)
    all_accuracies.append(for_gamma)

accuracies = pd.DataFrame(np.array(all_accuracies))
print(accuracies)
accuracies.to_csv('ACCURACIES_GAMMA_SPLIT.csv')
_ = sns.heatmap(accuracies)
_ = plt.show()

accuracies = pd.read_csv('ACCURACIES_GAMMA_SPLIT.csv')
accuracies = accuracies.infer_objects().drop('Unnamed: 0', axis=1)

x_ticks = [10, 15, 20, 25, 30, 35, 40, 45]
heatmap(accuracies, 'Impact of Changing Train-Test Split and Gamma on Label Propagation Accuracy', 'Test Size (%)', 'Gamma', x_ticks)



# Use the heatmap to determine the best train-test split and gamma value for Label Propagation
# Examine the performance of the optimal Label Propagation model
test_size = 0.15
gamma = 30
(x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled, x_train_lab_unlab, y_train_lab_unlab) = prepare_dataset(complete_dataset, test_size)
model = LabelPropagation(max_iter=1000, kernel='rbf', gamma=gamma)
# Fit model on our training dataset with labeled and unlabeled examples
model.fit(x_train_lab_unlab, y_train_lab_unlab)
# Make predictions on the test set
y_predictions = model.predict(x_test_labeled)
# Calculate accuracy of predictions on the test set
accuracy = accuracy_score(y_test_labeled, y_predictions)
print('Accuracy of Label Propagation Model on Test Data: %.3f' % (accuracy * 100) + '%')
# Print out the classification report and confusion matrix
conf_matrix = confusion_matrix(y_test_labeled, y_predictions, labels=model.classes_)
print('Classification Report')
print(classification_report(y_test_labeled, y_predictions))
print('Confusion Matrix')
print(conf_matrix)
# Verify that applying the label propagation model on the training data will lead to 100% accuracy (the class assignments are typically exactly correct for the training data)
model_check = model.predict(x_train_labeled)
model_check_accuracy = accuracy_score(y_train_labeled, model_check)
print('Accuracy of Label Propagation Model on Labeled Training Data: %.3f' % (model_check_accuracy * 100) + '%')



# Identify misclassified examples to better understand types of patients for which the model misclassifies
misclassified = np.where(y_test_labeled != y_predictions)
misclassified = misclassified[0]
misclassified_dataset = x_test_labeled[misclassified, :]
complete_dataset = np.array(complete_dataset)
for col in range(misclassified_dataset.shape[1]):
    # fig,a = plt.subplots(1, 2)
    # a[0].hist(misclassified_dataset[:, col], color='red', label='misclassified')
    # a[1].hist(complete_dataset[:, col], color='green', label='complete', sharex=a)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax1.hist(misclassified_dataset[:, col], color='red', label='misclassified')
    ax2.hist(complete_dataset[:, col], color='green', label='complete')
    plt.legend()
    plt.savefig('complete' + str(col) + '.png')
    plt.clf()
