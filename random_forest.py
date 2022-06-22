# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:07:49 2022

@author: dgarcialorenzo
"""


import pandas as pd

idir = r"C:\cpp\peanalysis\data\demandeEmploi_dfs_old"

df = pd.read_pickle(f"{idir}/compilation_months.pkl")
print(df.info())

df['ROME2'] = df.ROME_PROFESSION_CARD_CODE.replace(r'\w{2}$','',regex=True)
df.MONTH.map({1:"JAN",2:"FEV",3:"MAR",4:"APR",5:"MAY",6:"JUIN",
              7:"JUIL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"})
df.GENDER_CODE = df.GENDER_CODE.astype(int) -1

print(df.GENDER_CODE.unique())
# 0 Homme, 1 Femme

from sklearn.preprocessing import OneHotEncoder
drop_first = True
if drop_first:
    one_hot = OneHotEncoder(drop='first')
    
    encoded = one_hot.fit_transform(df[['AGE_GROUP_NAME']])
    df[one_hot.categories_[0][1:]] = encoded.toarray()
    
    
    encoded = one_hot.fit_transform(df[['DEPARTEMENT_NAME']])
    df[one_hot.categories_[0][1:]] = encoded.toarray()
    
    encoded = one_hot.fit_transform(df[['ROME2']])
    df[one_hot.categories_[0][1:]] = encoded.toarray()
else:
    one_hot = OneHotEncoder()
    
    encoded = one_hot.fit_transform(df[['AGE_GROUP_NAME']])
    df[one_hot.categories_[0]] = encoded.toarray()
    
    
    encoded = one_hot.fit_transform(df[['DEPARTEMENT_NAME']])
    df[one_hot.categories_[0]] = encoded.toarray()
    
    encoded = one_hot.fit_transform(df[['ROME2']])
    df[one_hot.categories_[0]] = encoded.toarray()
    
# encoded = one_hot.fit_transform(df[['REGISTRATION_CATEGORY_CODE']])
# df[one_hot.categories_[0]] = encoded.toarray()


# 'GENDER_CODE'

##  KEEPING TO PROCESS
# ROME2
# 'DEPARTEMENT_NAME'

# 'REGISTRATION_CATEGORY_CODE' ?????
y = df['MONTHS']

X = df.drop(columns=['REGION_CODE', 'ROME_PROFESSION_CARD_NAME', 'REGION_NAME',
       'REGISTRATION_CATEGORY_CODE', 'QUALIFICATION_NAME',
       'ROME_PROFESSION_CARD_CODE', 'DEPARTEMENT_NAME', 'GENDER_NAME',
       'DEPARTEMENT_CODE', 'AGE_GROUP_NAME', 'IDENT',
       'AGE_GROUP_CODE', '_id', 'QUALIFICATION_CODE','ROME2',"MONTHS"])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)

forest = RandomForestClassifier(n_estimators=100, random_state=100)
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

# Returns: 0.331 + WITHOUT REGISTRATION_CAT
# Returns: 0.35 +  WITH 