import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)
    
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("/workspace/goorm/titanic/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Survived', id_name='PassengerId')

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# print(X_train.info())
# print(y_train.head(5))
# print(y_train['Survived'].value_counts())
obj = ['Pclass','Sex','SibSp','Parch']
y = y_train['Survived']

X = pd.get_dummies(X_train[obj])
test = pd.get_dummies(X_test[obj])

# print(X.shape, test.shape)

from sklearn.ensemble import RandomForestClassifier
model =  RandomForestClassifier(n_estimators=200, max_depth=7, random_state=2022)
model.fit(X, y)
pred = model.predict(test)

# print(pred)
print(model.score(X, y))

output = pd.DataFrame({'PassengerId':X_test['PassengerId'], 'Survived': pred})
print(output.head)

output.to_csv('00000.csv',index=False)
check = pd.read_csv('00000.csv')
print(check.head())
print(model.score(test, y_test['Survived']))