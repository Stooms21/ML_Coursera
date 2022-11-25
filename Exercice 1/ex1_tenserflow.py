import pandas as pd
from sklearn import datasets
import tensorflow as tf
import itertools


FEATURES = ["sq-ft", "br"]
LABEL = ["price"]

data = pd.read_csv('../ex1data2.txt', sep=",", header=None)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols,
        model_dir="train")
