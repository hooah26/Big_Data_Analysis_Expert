import pandas as pd
import numpy as np

df =  pd.DataFrame({'user':['네모','네모',
                           '세모','세모'],
                   'Max Speed':[129., 240., 57., 41.]})

print(df)

print(df.groupby(['user']).mean())
