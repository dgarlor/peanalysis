import pandas as pd
import json
import os
from collections import defaultdict
# data = json.load(open(ifile))
# p = pd.DataFrame(data["result"]["records"])
# print(p)

idir = r"C:\cpp\peanalysis\data\demandeEmploi"

odir = idir+"_dfs"
os.makedirs(odir,exist_ok=True)

files = os.listdir(idir)
rp = defaultdict(list)

files.sort()
res = []


saved_dfs = os.listdir(odir)


for pe_source in files:
    print(pe_source)
    fdir = idir+os.sep+pe_source
    if not os.path.isdir(fdir):
        continue
    fdir = idir+os.sep+pe_source

    year = int(pe_source[-4:])
    month = int(pe_source[-6:-4])
    
    if year < 2000:
        # modified ones  MONTH YEAR
        year = int(pe_source[-6:-2])
        month = int(pe_source[-2:])

    
    rps = [fdir+os.sep+f for f in os.listdir(fdir) if f.endswith("json")]
    
    nrp = len(rps)
    print(year,month,pe_source,nrp)
    ofile = "df_%d_%02d_%s_n%d.pkl"%(year,month,pe_source,nrp)
    if ofile in saved_dfs:
        print(" already found! ",ofile)
        continue
        
    records = []
    for rdata in rps:
        x = json.load(open(rdata))
        records.extend(x["result"]["records"])
    
    df = pd.DataFrame(records)
    df.to_pickle(f"{odir}/{ofile}")
        
# a = pd.read_pickle(odir+"/df_2015_5_STMT_DEMANDE_052015_e85d10a4-78b0-4d37-bc25-f59d8fd33a21_n4859.pkl")
# b = pd.read_pickle(odir+"/df_2015_6_STMT_DEMANDE_062015_161c60c0-c9b9-44a1-81e0-94746158b732_n4807.pkl")

# ## GENERAL PROFESSION CODE, removing 2 last digits
# a.ROME2 = a.ROME_PROFESSION_CARD_CODE
# a.ROME2 = a.ROME2.replace(r'\w{2}$','',regex=True)
    