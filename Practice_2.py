import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
# print(df)
X_train, X_test, Y_train, Y_test = train_test_split(df, df['survived'], test_size=0.2, random_state=42,
                                                    stratify=df['survived'])
X_train = X_train.drop(['alive', 'survived'], axis=1)
X_test = X_test.drop(['alive', 'survived'], axis=1)

# print(X_train['embark_town'].value_counts())
# print(X_train['embarked'].value_counts())
# print(X_train['deck'].value_counts())


# 1. 결측치 제거
missing = ['age']
for i in missing:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_test[i].mean())

X_train['embark_town'] = X_train['embark_town'].fillna('Southampton')
X_test['embark_town'] = X_test['embark_town'].fillna('Southampton')
X_train['embarked'] = X_train['embarked'].fillna('S')
X_test['embarked'] = X_test['embarked'].fillna('S')
X_train['deck'] = X_train['deck'].fillna('C')
X_test['deck'] = X_test['deck'].fillna('C')

# print(X_train.isna().sum())
# print(X_test.isna().sum())


# 2.라벨인코딩
from sklearn.preprocessing import LabelEncoder

# print(X_train.info())

label = ['sex', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alone']
X_train[label] = X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] = X_test[label].apply(LabelEncoder().fit_transform)

# print(X_train.info())


# 3.데이터타입변환, 더미
dtype = ['pclass', 'sex', 'class']

for i in dtype:
    X_train[i] = X_train[i].astype('category')
    X_test[i] = X_test[i].astype('category')
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# print(X_train.head())

# 4.파생변수
X_train['age_qcut'] = pd.qcut(X_train['age'], 5, labels=False)
X_test['age_qcut'] = pd.qcut(X_test['age'], 5, labels=False)
# print(X_train.head())

# 5.스케일
from sklearn.preprocessing import MinMaxScaler

scaler = ['age', 'fare']
min = MinMaxScaler()
min.fit(X_train[scaler])
min.fit(X_test[scaler])
X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])
# print(X_train.head())


# 6.데이터분리
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42,
                                                      stratify=Y_train)
print(X_train.shape)
print(X_valid.shape)



