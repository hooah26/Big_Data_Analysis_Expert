import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic2.csv', parse_dates=['Date'], index_col=0)
print(df.head())


df_w = df.resample('W').sum()
print(df_w.head())

ma = df_w['Sales'].max()
ma