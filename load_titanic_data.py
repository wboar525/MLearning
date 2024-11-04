
import pandas as pd
import seaborn as sns


def load_titanic_data():
    data = sns.load_dataset('titanic')
    data.drop(['alive','embarked','class','sex'], inplace = True, axis=1)
    data["deck"]=data["deck"].cat.add_categories(["Unknown"])
    data["deck"]=data["deck"].fillna("Unknown")
    data['embark_town'] = data['embark_town'].fillna(data['embark_town'].mode()[0])
    data['age'] = data['age'].fillna(data['age'].mean())
    mean_adult_age = data[data["who"] != "child"]["age"].mean()
    data['age'] = data['age'].fillna(mean_adult_age)
    data_ohe = pd.get_dummies(data, dtype = int)
    data_ohe['adult_male']= data_ohe['adult_male'].astype(int)
    data_ohe['alone']= data_ohe['alone'].astype(int)
    y = data_ohe['survived']
    x = data_ohe.drop(['survived'], axis=1)
    return x,y



