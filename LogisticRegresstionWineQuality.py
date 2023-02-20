# import library
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("Datasets/WineQT.csv")
df.head(5)
df.info()

df['quality'].value_counts()

df = df.drop(columns = ['Id'])
df.head(5)

df.isnull().sum()

df['fixed acidity'].hist()
df['volatile acidity'].hist()
df['free sulfur dioxide'].hist()
df['total sulfur dioxide'].hist()

df.corr()
le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
df.head(100)

X = df.drop(columns = ['quality'])
Y = df['quality']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

model = LogisticRegression()
model.fit(X_train, Y_train)

print("Accuracy: ", model.score(X_test, Y_test) * 100)