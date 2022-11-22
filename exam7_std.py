import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
print(df.head)

enfj =  df[df['f4']=='ENFJ']['f1'].std()
# print(enfj
infp = df[df['f4']=='INFP']['f1'].std()
print(abs(enfj-infp))
