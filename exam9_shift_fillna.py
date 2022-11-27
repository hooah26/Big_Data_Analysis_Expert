import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic2.csv')
print(df.head())


df['previous_PV'] = df['PV'].shift(1)
df.head()

df['previous_PV'] = df['previous_PV'].fillna(method = 'bfill')
df.head()


cond = (df['Events'] == 1) & (df['Sales'] <= 1000000)
print(df[cond]['previous_PV'].sum())