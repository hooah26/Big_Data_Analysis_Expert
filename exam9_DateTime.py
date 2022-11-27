import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic2.csv')

# print(df.head())

df['Date'] = pd.to_datetime(df['Date'])
# print(df.info())
# print(df.head())

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# print(df.head())

cond = (df['year'] ==2022) & (df['month']==5)
print(df[cond]['Sales'].median())