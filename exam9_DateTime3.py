import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic2.csv', parse_dates = ['Date'])
# print(df.head())

df.info()

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

# print(df)

def event_sales(x):
    if x['Events'] == 1:
        x['Sales2'] = x['Sales']*0.8
        
    else :
        x['Sales2'] = x['Sales']
    return x

df =df.apply(lambda x : event_sales(x), axis = 1)
# print(df.head())


cond = df['year'] == 2022
df1 = df[cond]
print(df1)
sale1 = df1.groupby('month')['Sales2'].sum().max()
print(sale1)
int(round(abs(sale1 - sale2),0))
