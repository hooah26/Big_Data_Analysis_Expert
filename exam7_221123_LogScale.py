import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/HousePrices/train.csv')

print(df['SalePrice'].hist())