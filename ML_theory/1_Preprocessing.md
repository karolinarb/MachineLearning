## Data preprocessing tools

# Importing libraries
<import numpy as np
import matplotlib.pyplot as plt
import pandas as pd>

# Importing the dataset
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values>
```
# Missing data strategies
1. Dropping
2. Mean replacement
    - Replace missing values with the mean value from the rest of the column (columns, not rows! 
      A column represents a single feature; it only makes sense to take the mean from other
      samples of the same feature.)
    - Fast & easy, won’t affect mean or sample size of overall data set
    - Median may be a better choice than mean when outliers are present
    - Only works on column level, misses correlations between features
    - Can’t use on categorical features (imputing withmost frequent value can work in this case, though)
    - Not very accurate
 
4. Machine Learning:
    - KNN
    - Deep Learning 
    - Regression
    - 
