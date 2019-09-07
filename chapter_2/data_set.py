import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display


X, y = mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
# print("X.shape: {}".format(X.shape))  # X.shape: (26, 2)
if __name__ == "__main__":
  plt.show() 


X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature") # 特徴量
plt.ylabel("Target")
if __name__ == "__main__":
  plt.show()


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# print("cancer.keys():\n{}".format(cancer.keys()))

"""
cancer.keys():
dict_keys(['target_names', 'target', 'filename', 'feature_names', 'DESCR', 'data'])

"""


# print("Shape of cancer data: {}".format(cancer.data.shape))

"""
Shape of cancer data: (569, 30)

"""

# print("Sample counts per class:\n{}".format(
#   {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
# ))

"""
Sample counts per class:
{'malignant': 212, 'benign': 357}

"""

# print("Feature names:\n{}".format(cancer.feature_names))

"""
Feature names:
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
 
"""

from sklearn.datasets import load_boston

boston = load_boston()
# print("Data shape: {}".format(boston.data.shape))

"""
Data shape: (506, 13)

"""

X, y = mglearn.datasets.load_extended_boston()
# print("X.shape: {}".format(X.shape))

"""
X.shape: (506, 104)

"""
