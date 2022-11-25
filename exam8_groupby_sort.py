import pandas as pd
import numpy as np

df= pd.read_csv('/workspace/goorm/covid-vaccination-vs-death_ratio.csv')
# print(df)

df2 = df.groupby('country').max()
df2 = df2.sort_values(by='ratio', ascending = False)

df2 = df2[1:]
top = df2['ratio'].head(10).mean()
bottom = df['ratio'].tail(10).mean()
print(top, bottom)

# print(df2.head())
