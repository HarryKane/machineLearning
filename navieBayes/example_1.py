import pandas as pd
import numpy as np
from sklearn import cross_validation
from MultinomialNB import MultinomialNB
from GaussianNB import GaussianNB


data = pd.read_csv('../data/train.csv')
data = data.replace(np.nan,-1)
for col in ['Pclass','Age','SibSp','Parch','Fare']:
    col_max = data[col].max()
    col_min = data[col].min()
    data[col] = (data[col] - col_min) / (col_max - col_min)
data = data[['Pclass','Age','SibSp','Parch','Fare','Sex','Survived']]
data['Sex_male'] = data['Sex'].apply(lambda x: 1 if x=='male' else 0)
data['Sex_female'] = data['Sex'].apply(lambda x: 1 if x=='female' else 0)
train, test = cross_validation.train_test_split(data, test_size=0.3, random_state=2018)
train_y = train['Survived']
train_x = train[['Pclass','Age','SibSp','Parch','Fare','Sex_male','Sex_female']]
test_y = test['Survived']
test_x = test[['Pclass','Age','SibSp','Parch','Fare','Sex_male','Sex_female']]




#clf = MultinomialNB()
clf = GaussianNB()

clf.fit(train_x,train_y)
acc = clf.test(test_x,test_y)
print(acc)