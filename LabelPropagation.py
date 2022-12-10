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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation


# Load in the data
train_path = os.path.join('.', 'Core_Dataset_SEMI-SUPERVISED.csv')
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

# Split the labeled data into train and test set using a 70-30 split
x_train_labeled, x_test_labeled, y_train_labeled, y_test_labeled = train_test_split(labeled_x, labeled_y, test_size=0.30, random_state=1, stratify=labeled_y)

# Prepare unlabeled data for semi-supervised model
unlabeled_data = complete_dataset[complete_dataset['PRED'].isnull()]
unlabeled_x = np.array(unlabeled_data.drop('PRED', axis=1))
# The labels for the unlabeled data should be -1 for LabelPropagation algorithm
unlabeled_y = np.full(unlabeled_x.shape[0], fill_value=-1)

# Finally create the training datasets with both labeled and unlabeled examples
x_train_lab_unlab = np.concatenate((x_train_labeled, unlabeled_x))
y_train_lab_unlab = np.concatenate((y_train_labeled, unlabeled_y))






# Implementation of Label Propagation Model
model = LabelPropagation(max_iter=50000)
# Fit model on our training dataset with labeled and unlabeled examples
model.fit(x_train_lab_unlab, y_train_lab_unlab)
# Make predictions on the test set
y_predictions = model.predict(x_test_labeled)
# Calculate accuracy of predictions on the test set
accuracy = accuracy_score(y_test_labeled, y_predictions)
print('Accuracy of Label Propagation Model: %.3f' % (accuracy * 100) + '%')





# Compare LabelPropagation Model to Baseline Logistic Regression Model
# Implement Logistic Regression Model on labeled data
baseline_model = LogisticRegression()
baseline_model.fit(x_train_labeled, y_train_labeled)
baseline_y_predictions = baseline_model.predict(x_test_labeled)
# Calculate accuracy of predictions on the test set for the baseline logistic regression model
baseline_accuracy = accuracy_score(y_test_labeled, baseline_y_predictions)
print('Accuracy of Baseline Logistic Regression Model: %.3f' % (baseline_accuracy * 100) + '%')





# We can extend this semi-supervised model by using the inferred labels for the unlabeled data
# We can then feed in this data as well for a supervised model

# Grab labels for the full training dataset (includes labeled data and newly inferred labels for originally unlabeled data)
full_training_labels = model.transduction_
# Implement Logistic Regression as the supervised model on the full training dataset
supervised_model = LogisticRegression(max_iter=1000)
supervised_model.fit(x_train_lab_unlab, full_training_labels)
supervised_model_predictions = supervised_model.predict(x_test_labeled)
# Calculate accuracy of supervised model
supervised_accuracy = accuracy_score(y_test_labeled, supervised_model_predictions)
print('Accuracy of Supervised Model: %.3f' % (supervised_accuracy * 100) + '%')

