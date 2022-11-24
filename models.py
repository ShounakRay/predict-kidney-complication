# @Author: shounak
# @Date:   2022-11-22T23:18:49-08:00
# @Email:  shounak@stanford.edu
# @Filename: models.py
# @Last modified by:   shounak
# @Last modified time: 2022-11-22T23:19:14-08:00

from sklearn.linear_model import LinearRegression

_ = """
################################################################################
################################### TRAINING ##################################
"""

train_size = int(len(JOINED_FINAL) * 0.7)
reduced = JOINED_FINAL.sample(frac=1).reset_index(drop=True)
training_set = reduced.values[:train_size, :]
test_set = reduced.values[train_size:, :]

model = LinearRegression()
x = training_set[:, :-1]
y = training_set[:, -1]
model.fit(x, y)
model.score(x, y)

model.predict(test_set[:, :-1])
test_set[:, -1]
