# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:20:22 2022

@author: dgarcialorenzo
"""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

datadir = "stream_data"

st.write("""# Estimation du nombre de demandeurs d'emploi par deep learning""")
st.write(" Ce projet explore la possibilité de estimer le nombre de demandeurs d'emploi dans le mois prochain en utilisant seulement le nombre de demandeurs d'emploi du mois actuel")

st.write("- Pour plus d'information sur les donneés: https://pole-emploi.io/data/api/infotravail?tabgroup-api=documentation&doc-section=api-doc-section-caracteristiques")
st.write("- Le référentiel utilisé est 'Statistiques sur le marché du travail' ")


st.write("## Données")


def loadJsonData(jsonfile):
    x = json.load(open(jsonfile))
    return pd.DataFrame(x["result"]["records"])

## READING DATA
df = pd.read_pickle(f"{datadir}/sagg_Demandeurs2012.2016.pkl")
regions = df.REGION_NAME.unique()
categories = df.REGISTRATION_CATEGORY_CODE.unique()

requete = loadJsonData( f"{datadir}/rp_141a3516-dd5a-49ad-9887-feacf9b64456_o0.json")

st.write("Les colonnes disponibles* dans l'api sont les suivantes:")
st.write(requete.columns)
st.write("_* La colonne IDENT n'est pas disponible qu'à partir 2015._")

st.write("### Selection de données: 2012 - 2016")
st.write("""Pour cet étude préliminaire on a limité les données utilisées par manque de temps ou des problèmes sur l'api:
- Quelques mois en 2017 et 2019 semblent incompletes (taille de données est significativament inférieur aux autres mois). 
- Le changment de régions aurait obligé à recomposer les régions à partir des départements. On s'est limité à : %s
""" % (", ".join(regions)))


st.write(df.head(5))

# Plot figure
###################################################################

if st.checkbox("Montrer les demandeurs d'emploi par région"):
    selectedRegion = st.selectbox("Sélectionner une région:", regions)
    
    rdf = df[df.REGION_NAME == selectedRegion]
    fig, ax = plt.subplots(layout='constrained')
    for c in categories:  
        cdf = rdf[df.REGISTRATION_CATEGORY_CODE == c].sort_values(by="DATE")
        ax.plot(cdf.DATE, cdf.DEMANDEURS, label=c)  # Plot some data on the axes.
    ax.set_title("Demandeurs en "+selectedRegion)  # Add a title to the axes.
    ax.legend();  # Add a legend.
    st.write(fig)
if st.checkbox("Montrer les demandeurs d'emploi par catégorie"):
    selectedCategory = st.selectbox("Sélectionner une categorie:", categories)
    
    cdf = df[df.REGISTRATION_CATEGORY_CODE == selectedCategory]
    fig, ax = plt.subplots(layout='constrained')
    for r in regions:  
        rdf = cdf[df.REGION_NAME == r].sort_values(by="DATE")
        ax.plot(rdf.DATE, rdf.DEMANDEURS, label=r)  # Plot some data on the axes.
    ax.set_title("Demandeurs "+selectedCategory)  # Add a title to the axes.
    ax.legend();  # Add a legend.
    st.write(fig)
    
    
st.write("## Modèle apprentissage simple")
st.write(""" Nous allons proposer un première modèle avec des couches Dense
- Entrées: MONTH  REGION_NAME  REGISTRATION_CATEGORY_CODE  DEMANDEURS(N)
- Sortie: DEMANDEURS(N+1)
    """)


demandeur_mean = np.mean(train_data, axis=0)[-2]
demandeur_std = np.std(train_data, axis=0)[-2]
model = firstmodel(use_embeddings, regions, categories,
                   demandeur_mean, demandeur_std)
try:
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
except:
    print(" -- Error loading model graph")
