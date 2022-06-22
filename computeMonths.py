# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:50:26 2022

@author: dgarcialorenzo
"""

import pandas as pd
import json
import os
from collections import defaultdict
from copy import deepcopy

# data = json.load(open(ifile))
# p = pd.DataFrame(data["result"]["records"])
# print(p)

idir = r"C:\cpp\peanalysis\data\demandeEmploi_dfs_old"
pdfiles = os.listdir(idir)
pdfiles.sort(key = lambda x:(int(x.split("_")[1]),int(x.split("_")[2])))


afile = ""
bfile = ""
for i,p in enumerate(pdfiles):
    print(i,p)
    if p.count("2015_1_"):
        afile = p
    if p.count("2015_2_"):
        bfile = p
        
def readDF(filename):
    return pd.read_pickle(idir+os.sep+filename)
def addYearMonth(df, filename):
    sp = filename.split("_")
    df["YEAR"] = int(sp[1])
    df["MONTH"] = int(sp[2])
    

def readDF2(year,month):
    for p in pdfiles:
        if p.count("%d_%d_"%(year,month)):
            df = readDF(p)
            df["YEAR"] = year
            df["MONTH"] = month
            return 
    print("Not found",year,month)
    return None
a = pd.read_pickle(idir+os.sep+afile)
b = pd.read_pickle(idir+os.sep+bfile)

# ## GENERAL PROFESSION CODE, removing 2 last digits
# a.ROME2 = a.ROME_PROFESSION_CARD_CODE
# a.ROME2 = a.ROME2.replace(r'\w{2}$','',regex=True)
selected = pdfiles[26:47]


done = []

s0 = readDF(selected[0])
addYearMonth(s0,selected[0])
# setting to -1000 to filter them out
running = deepcopy(s0) # running are all initials!
running.insert(s0.shape[1],"MONTHS",-1000) # initials set to negatif to filter them out

for sname in selected[1:]:
    print(" -- CHECKING: ",sname)
    # read new month
    s1 = readDF(sname)    
    addYearMonth(s1, sname)
    
    # Those not in new month
    stillRunning = running.IDENT.isin(s1.IDENT.values)
    
    stopping = running[ stillRunning == False]
    # storing them outside
    done.append(stopping[stopping.MONTHS >= 0])
    print("STOPPING: ",stopping.shape[0])
    print("DONE: ", len(done))
    # Remaining running add 1
    running = running[ stillRunning ]
    running.MONTHS += 1
    print("CONTINUE: ",running.shape[0])
    
    # addint new lines not in s0
    newbies = s1[s1.IDENT.isin(s0.IDENT.values)  == False]
    newbies.insert(newbies.shape[1],"MONTHS",0) # initial as zero and counting
    print("NEWBIES: ",newbies.shape[0])
    
    running = pd.concat([running,newbies])
    print("RUNNING: ",running.shape[0])
    s0 = s1

# Concatenate all steps
counts  = pd.concat(done)
counts = counts[counts.MONTHS >= 0]
counts.to_pickle(f"{idir}/compilation_months.pkl")

p201601 = readDF2(2017,1)
print(p201601.columns)


## DOING REGRESSION
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(drop='first')
counts['ROME2'] = a.ROME_PROFESSION_CARD_CODE.replace(r'\w{2}$','',regex=True)

rome =pd.DataFrame( enc.fit_transform(counts[['ROME2']]).toarray())
age = pd.DataFrame(enc.fit_transform(counts[['AGE_GROUP_CODE']]).toarray())
dept = pd.DataFrame(enc.fit_transform(counts[['DEPARTEMENT_CODE']]).toarray())
gender = pd.DataFrame(enc.fit_transform(counts[['GENDER_CODE']]).toarray())
months = pd.DataFrame(enc.fit_transform(counts[['MONTH']]).toarray())
xdata = pd.concat([rome,age,gender,months],axis=1)
ydata = counts.MONTHS


model = LinearRegression()
scores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(xdata, ydata)):
 model.fit(xdata.iloc[train], ydata.iloc[train])
 score = model.score(xdata.iloc[test], ydata.iloc[test])
 scores.append(score)
 
print(scores)


from sklearn import svm
clf = make_pipeline(StandardScaler(), SVC())
scores = []
for i, (train, test) in enumerate(kfold.split(xdata, ydata)):
 clf.fit(xdata.iloc[train], ydata.iloc[train])
 score = clf.score(xdata.iloc[test], ydata.iloc[test])
 scores.append(score)
print(scores)
