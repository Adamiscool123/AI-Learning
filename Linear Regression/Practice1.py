import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "winequality-red.csv")

data_set = pd.read_csv(file_path)

sugar = data_set["residual sugar"]

pH = data_set["pH"]

x = sugar

y = pH

X_train, X_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2,       # Reserves 20% of data for testing
    random_state=42,     # Ensures the split is reproducible
)

b = 0

w = 0

k = 10000

for i in range(0, k):

    y1 = w*(X_train)+b

    errors = y1-y_train

    length_of_errors = errors.size

    MSE = (np.sum((errors)**2)/length_of_errors)

    total_sum_w = np.sum(errors*X_train)

    total_sum_b = np.sum(errors)

    gradient_w = total_sum_w/length_of_errors

    gradient_b = total_sum_b/length_of_errors

    a = 0.01

    w = w-(a*gradient_w)

    b = b-(a*gradient_b)

y2 = w*(0.076)+b

print(w)
print(b)
print(y2)