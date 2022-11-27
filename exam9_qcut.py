import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
# print(df.head())

print(df.shape)
df = df[~(df['age'] <= 0)]
print(df.shape)
df = df[(df['age'] == round(df['age'], 0))]
print(df.shape)

df['range'] = pd.qcut(df['age'], q=3, labels=['group1','group2','group3'])
print(df['range'].value_counts())
g1_med = df[df['range'] == 'group1']['age'].median()
g2_med = df[df['range'] == 'group2']['age'].median()
g3_med = df[df['range'] == 'group3']['age'].median()

print(g1_med + g2_med + g3_med)
