# Import Library
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# importing dataset
df = pd.read_csv("Datasets/Iris.csv")

df.head(5)
df.info()
df['Species'].value_counts()

df = df.drop(columns = ['Id'])
df.head(5)
df.isnull().sum()

df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()

df.corr()

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head(100)

X = df.drop(columns = ['Species'])
Y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
model = LogisticRegression()

model.fit(X_train, Y_train)

print("Accuracy: ", model.score(X_test, Y_test) * 100)