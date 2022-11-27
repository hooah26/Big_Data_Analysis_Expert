import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
# print(df)

df =  df.groupby(['city','f4'])[['f5']].mean()
# print(df)

df = df.reset_index().sort_values('f5', ascending=False).head(7)
print(df)

print(round(df['f5'].sum(),2))