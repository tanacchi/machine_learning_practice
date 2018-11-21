# page 14

# ===============================
#  Getting and checking dataset
# ===============================

from sklearn.datasets import load_iris

iris_dataset = load_iris()
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

"""
Keys of iris_dataset: 
dict_keys(['data', 'DESCR', 'filename', 'target_names', 'feature_names', 'target'])

"""


# print(iris_dataset['DESCR'][:193] + "\n ...")

"""
Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, pre
 ...

"""


# print("Target names: {}".format(iris_dataset['target_names']))

"""
Target names: ['setosa' 'versicolor' 'virginica']

"""


# print("Feature names: \n{}".format(iris_dataset['feature_names']))

"""
Feature names: 
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

"""


# print("Type of data: {}".format(type(iris_dataset['data'])))
# print("Shape of data: {}".format(iris_dataset['data'].shape))

"""
Type of data: <class 'numpy.ndarray'>
Shape of data: (150, 4)

"""


# print("First five column of data:\n{}".format(iris_dataset['data'][:5]))

"""
First five column of data:
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]

"""


# print("Type of target: {}".format(type(iris_dataset['target'])))

"""
Type of target: <class 'numpy.ndarray'>

"""


# print("Shape of target: {}".format(iris_dataset['target'].shape))

"""
Shape of target: (150,)

"""

print("Target:\n{}".format(iris_dataset['target']))

# setona: 0, versicolor: 1, verginia: 2
"""
Target:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

"""

