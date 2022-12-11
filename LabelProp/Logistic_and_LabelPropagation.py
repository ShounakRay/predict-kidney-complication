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


def prepare_dataset(complete_dataset):
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

    # Split the labeled data into train and test set using a 60-40 split
    # Also experiment with other test-train splits by adjusting test_size
    x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled = train_test_split(labeled_x, labeled_y,
                                                                                        test_size=0.40, random_state=1,
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




x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled, x_train_lab_unlab, y_train_lab_unlab = prepare_dataset(complete_dataset)






# Baseline Logistic Regression Model
# Implement Logistic Regression Model on labeled data
# Experiment with L1 vs L2 regularization by changing the penalty
baseline_model = LogisticRegression(penalty='l1', max_iter=1000, solver='saga')
baseline_model.fit(x_train_labeled, y_train_labeled)
baseline_y_predictions = baseline_model.predict(x_test_labeled)
# Calculate accuracy of predictions on the test set for the baseline logistic regression model
baseline_accuracy = accuracy_score(y_test_labeled, baseline_y_predictions)
print('Accuracy of Baseline Logistic Regression Model on Test Data: %.3f' % (baseline_accuracy * 100) + '%')
# Print out the classification report and confusion matrix
baseline_conf_matrix = confusion_matrix(y_test_labeled, baseline_y_predictions, labels=baseline_model.classes_)
print('Classification Report')
print(classification_report(y_test_labeled, baseline_y_predictions))
print('Confusion Matrix')
print(baseline_conf_matrix)

# Check if model is overfitting
baseline_overfitting_check = baseline_model.predict(x_train_labeled)
baseline_overfitting_check_accuracy = accuracy_score(y_train_labeled, baseline_overfitting_check)
print('Accuracy of Baseline Logistic Regression Model on Labeled Training Data: %.3f' % (baseline_overfitting_check_accuracy * 100) + '%')
# Print out the classification report and confusion matrix for the model on the training data
baseline_conf_matrix = confusion_matrix(y_train_labeled, baseline_overfitting_check, labels=baseline_model.classes_)
print('Classification Report')
print(classification_report(y_train_labeled, baseline_overfitting_check))
print('Confusion Matrix')
print(baseline_conf_matrix)



# # Attempt to preprocess data to improve model performance for baseline logistic regression
# # Remove unimportant features from the dataset
# complete_dataset = reduced_dataset(complete_dataset, model=baseline_model, REMOVE_THIS_PERCENT=85, plot=False)
# x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled, x_train_lab_unlab, y_train_lab_unlab = prepare_dataset(complete_dataset)
# # Implement baseline Logistic Regression Model on reduced dataset with fewer features
# print("REDUCED DATASET:")
# baseline_model = LogisticRegression(penalty='l1', max_iter=1000, solver='saga')
# baseline_model.fit(x_train_labeled, y_train_labeled)
# baseline_y_predictions = baseline_model.predict(x_test_labeled)
# # Calculate accuracy of predictions on the test set for the baseline logistic regression model
# baseline_accuracy = accuracy_score(y_test_labeled, baseline_y_predictions)
# print('Accuracy of Baseline Logistic Regression Model on Test Data: %.3f' % (baseline_accuracy * 100) + '%')
# # Print out the classification report and confusion matrix
# baseline_conf_matrix = confusion_matrix(y_test_labeled, baseline_y_predictions, labels=baseline_model.classes_)
# print('Classification Report')
# print(classification_report(y_test_labeled, baseline_y_predictions))
# print('Confusion Matrix')
# print(baseline_conf_matrix)
# # Check if model is overfitting
# baseline_overfitting_check = baseline_model.predict(x_train_labeled)
# baseline_overfitting_check_accuracy = accuracy_score(y_train_labeled, baseline_overfitting_check)
# print('Accuracy of Baseline Logistic Regression Model on Labeled Training Data: %.3f' % (baseline_overfitting_check_accuracy * 100) + '%')



print('--------------------------------') # spacing



# Implementation of Label Propagation Model
# Experiment with kernel functions and gamma parameter
gammas = []
test_accuracies = []
train_accuracies = []
for gamma in np.arange(0.1, 150, 1):
    gammas.append(gamma)
    print(gamma)
    model = LabelPropagation(max_iter=100000, kernel='rbf', gamma=gamma)
    # Fit model on our training dataset with labeled and unlabeled examples
    model.fit(x_train_lab_unlab, y_train_lab_unlab)
    # Make predictions on the test set
    y_predictions = model.predict(x_test_labeled)
    # Calculate accuracy of predictions on the test set
    accuracy = accuracy_score(y_test_labeled, y_predictions)
    test_accuracies.append(accuracy)
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
    train_accuracies.append(model_check_accuracy)
    print('Accuracy of Label Propagation Model on Labeled Training Data: %.3f' % (model_check_accuracy * 100) + '%')



plt.plot(gammas, test_accuracies, label='Test')
plt.plot(gammas, train_accuracies, label='Train')
plt.xlabel('Gamma values')
plt.ylabel('Accuracy')
plt.title('Impact of Changing Gamma Values on Label Propagation Accuracy')
plt.legend()
plt.tight_layout()
#plt.figure(figsize=())
# plt.savefig(bbox_inches=True)
plt.show()


# From the plot generated above, we found that the best value for gamma is 30
# We now want to get the accuracy and confusion matrix for Label Propagation when gamma is 30
model = LabelPropagation(max_iter=100000, kernel='rbf', gamma=110)
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

