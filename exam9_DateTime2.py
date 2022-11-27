import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic2.csv', parse_dates = ['Date'])
# print(df.head())

df.info()

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

df.head()
df['weekend'] = df['dayofweek'].apply(lambda x: x>=5)

df.head()

weekend_cond = (df['year']==2022) & (df['month']==5) & (df['weekend'])
weekday_cond = (df['year']==2022) & (df['month']==5) & (~df['weekend'])

weekend = df[weekend_cond]['Sales'].mean()
weekday = df[weekday_cond]['Sales'].mean()

round(weekend - weekday, 1)