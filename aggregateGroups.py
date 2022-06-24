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

idir = r"C:\cpp\peanalysis\data\demandeEmploi_dfs"
pdfiles  = [p for p in os.listdir(idir) if p.startswith("df")]
pdfiles.sort(key = lambda x:(int(x.split("_")[1]),int(x.split("_")[2])))
for p in pdfiles:
    print(p)

selected = ['REGION_NAME', 'REGISTRATION_CATEGORY_CODE',
       'DEPARTEMENT_NAME', 'GENDER_NAME', 
       'AGE_GROUP_NAME','QUALIFICATION_CODE', 
       'YEAR', 'MONTH', 'ROME1', 'ROME2']

groupby = ['REGION_NAME', 'REGISTRATION_CATEGORY_CODE',
       'GENDER_NAME', 'YEAR','MONTH','AGE_GROUP_NAME']

def readDF(filename, load_only=False):
    df = pd.read_pickle(idir+os.sep+filename)
    if not load_only:
        sp = filename.split("_")
        df["YEAR"] = int(sp[1])
        df["MONTH"] = int(sp[2])
        df['ROME1'] = df.ROME_PROFESSION_CARD_CODE.replace(r'\w{4}$','',regex=True)
        df['ROME2'] = df.ROME_PROFESSION_CARD_CODE.replace(r'\w{2}$','',regex=True)
        df['DEMANDEURS'] = 1
        #df = df.drop(columns=[c for c in df.columns if c not in selected])
    return df


agg = []
for p in pdfiles:
    sp = p.split("_")
    if not  p.startswith("df"):
        continue
    year,month = sp[1:3]
    if year not in ["2012","2013","2014","2015","2016"]:
        continue
    
    df = readDF(p)
    gdf = df.groupby(by=groupby)["DEMANDEURS"].count().reset_index()
    print(year,month,"size:",gdf.shape)
    
    agg.append(gdf)


aggdf = pd.concat(agg).reset_index()
aggdf["day"]=1
aggdf["DATE"] = pd.to_datetime(aggdf[["YEAR","MONTH","day"]],yearfirst=True)
regions = ['Bretagne', 'Centre', 'Corse', 'Pays de la Loire',
       "Provence-Alpes-Côte d'Azur", 'Île-de-France']
aggdf = aggdf[aggdf.REGION_NAME.isin(regions)]
aggdf = aggdf[aggdf.REGISTRATION_CATEGORY_CODE != "1"]
aggdf = aggdf[aggdf.REGISTRATION_CATEGORY_CODE != "2"]

#  checking no nan
aggdf[df.isna().any(axis=1)]

aggdf = aggdf.groupby(by=['MONTH','DATE','REGION_NAME',"REGISTRATION_CATEGORY_CODE"])["DEMANDEURS"].sum().reset_index()
aggdf.to_pickle(idir+"agg_Demandeurs2012.2016.pkl")
# plot demandeurs
aggdf.plot.scatter("DATE","DEMANDEURS")

## PIVOTING TABLE
df = aggdf.pivot_table(values="DEMANDEURS", index=["DATE","MONTH"], columns=["REGION_NAME","REGISTRATION_CATEGORY_CODE"]).reset_index()
df.to_pickle(idir+"pivot_Date_vs_Category.pkl")
