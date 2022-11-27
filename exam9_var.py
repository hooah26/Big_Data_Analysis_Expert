import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')


cond=df['f2']==0
df =  df[cond].sort_values('age', ascending=True).reset_index(drop=True)
print(df)

df = df[:20]

df_var1 = df['f1'].var()
df['f1'] = df['f1'].fillna(df['f1'].min())
df_var2 = df['f1'].var()

print(round(df_var1-df_var2), 2)