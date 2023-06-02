## Data preprocessing tools

### Importing libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd>
```

### Importing the dataset

```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values>
```
x - independent variables
y - dependent variable

### Missing data strategies

1. Dropping
    - If not many rows contain missing data and dropping those rows doesn’t bias your data
    - But, it’s never going to be the right answer for the “best” approach.
    
2. Mean replacement
    - Replace missing values with the mean value from the rest of the column (columns, not rows! 
      A column represents a single feature; it only makes sense to take the mean from other
      samples of the same feature.)
    - Fast & easy, won’t affect mean or sample size of overall data set
    - Median may be a better choice than mean when outliers are present
    - Only works on column level, misses correlations between features
    - Can’t use on categorical features (imputing withmost frequent value can work in this case, though)
    - Not very accurate

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,a:b])
x[:,1:3]=imputer.transform(x[:,a:b])
```
 
3. Machine Learning:
    - KNN: Find K “nearest” (most similar) rows and average their values
            - Assumes numerical data, not categorical
    - Deep Learning:
            - Build a machine learning model to impute data for your machine learning model!
            - Works well for categorical data. Really well. But it’s complicated.
    - Regression:
            - Find linear or non-linear relationships between the missing feature and other features
            - Most advanced technique: MICE (Multiple Imputation by Chained Equations)
  
### Encoding categorical data

Categorical data:
    - Qualitative data that has no inherent mathematical meaning
    - Gender, Yes/no (binary data), Race, State of Residence, Product Category, Political Party, etc.
    - You can assign numbers to categories in order to represent them more compactly, but the numbers don’t have mathematical meaning

#### Encoding the Independent Variable

One-hot encoding:
    - Create “buckets” for every category
    - The bucket for your category has a 1, all others have a 0
    - Very common in deep learning, where categories are represented by individual output “neurons”
    
```python 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
```
e.g France, Germany, Spain -> 1.0.0, 0.1.0, 0.0.1  

#### Encoding the Dependent Variable

```python 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```
e.g No, Yes -> 0, 1

### Splliting dataset into train and test sets

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### Feature Scaling

Some models prefer feature data to be normally distributed around 0 (most neural nets). Most models require feature data to at least be scaled to comparable values. Otherwise features with larger magnitudes will have more weight than they should
Example: modeling age and income as features – incomes will be much higher values than ages
 
```python 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])
```
