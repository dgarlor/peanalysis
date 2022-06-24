# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:20:22 2022

@author: dgarcialorenzo
"""

from demandeurs_keras_model import firstmodel
from demandeurs_keras_preprocessing import prepareData
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

datadir = "stream_data"

st.write("""# Estimation du nombre de demandeurs d'emploi par deep learning""")
st.write(" Ce projet explore la possibilité d'estimer le nombre de demandeurs d'emploi pour le mois suivant en utilisant seulement le nombre de demandeurs d'emploi du mois actuel")
st.write("- Les sources se trouvent ici: https://github.com/dgarlor/peanalysis")
st.write("- Pour plus d'information sur les donneés: https://pole-emploi.io/data/api/infotravail?tabgroup-api=documentation&doc-section=api-doc-section-caracteristiques")
st.write("- Le référentiel utilisé est 'Statistiques sur le marché du travail' ")


st.write("## Données")


def loadJsonData(jsonfile):
    x = json.load(open(jsonfile))
    return pd.DataFrame(x["result"]["records"])


# READING DATA
df = pd.read_pickle(f"{datadir}/sagg_Demandeurs2012.2016.pkl")
regions = df.REGION_NAME.unique()
categories = df.REGISTRATION_CATEGORY_CODE.unique()

requete = loadJsonData(
    f"{datadir}/rp_141a3516-dd5a-49ad-9887-feacf9b64456_o0.json")

st.write("Les colonnes disponibles* dans l'api sont les suivantes:")
st.write(requete.columns)
st.write("_* La colonne IDENT n'est pas disponible qu'à partir 2015._")

st.write("### Selection de données: 2012 - 2016")
st.write("""Pour cette étude préliminaire on a limité les données utilisées par manque de temps ou d'autres problèmes:
- Quelques mois en 2017 et 2019 semblent incomplets (taille de données est significativament inférieur aux autres mois). 
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
        # Plot some data on the axes.
        ax.plot(cdf.DATE, cdf.DEMANDEURS, label=c)
    ax.set_title("Demandeurs en "+selectedRegion)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    st.write(fig)
if st.checkbox("Montrer les demandeurs d'emploi par catégorie"):
    selectedCategory = st.selectbox("Sélectionner une categorie:", categories)

    cdf = df[df.REGISTRATION_CATEGORY_CODE == selectedCategory]
    fig, ax = plt.subplots(layout='constrained')
    for r in regions:
        rdf = cdf[df.REGION_NAME == r].sort_values(by="DATE")
        # Plot some data on the axes.
        ax.plot(rdf.DATE, rdf.DEMANDEURS, label=r)
    ax.set_title("Demandeurs "+selectedCategory)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    st.write(fig)


st.write("## Modèle apprentissage simple")
st.write(""" Nous allons proposer un premier modèle avec des couches Denses
- Entrées: MONTH  REGION_NAME  REGISTRATION_CATEGORY_CODE  DEMANDEURS(N)
- Sortie: DEMANDEURS(N+1)

    """)

_, (test_x, test_y), demandeur_mean, demandeur_std = prepareData(df)

use_embeddings = st.checkbox(
    "Utilisation d'embeddings pour les entrée (par défaut one hot encoding)")
suffix = "emb" if use_embeddings else "oneh"
model = firstmodel(use_embeddings, regions, categories,
                   demandeur_mean, demandeur_std)
try:
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
except:
    #st.write(" -- Error loading model graph figure")
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    st.text("\n".join(stringlist))


st.write("## Apprentissage")

history = pd.read_pickle(f"{datadir}/learned_history_{suffix}.pkl")
history = history.drop(columns=["accuracy", "val_accuracy"])
history.plot.line()

fig, ax = plt.subplots(layout='constrained')
ax.plot(history["loss"], label="Apprentissage")  # Plot some data on the axes.
ax.plot(history["val_loss"], label="Validation")  # Plot some data on the axes.
ax.set_title("Evolution de la fonction de coût")  # Add a title to the axes.
ax.legend()  # Add a legend.
st.write(fig)

# Show difference
if True:
    model.load_weights(f"{datadir}/learned_model_{suffix}.keras")
    test_output = model.predict(test_x)
    xx = (test_output*demandeur_std)+demandeur_mean
    np.save(f"{datadir}/results_{suffix}.npy", xx)
else:
    xx = np.load(f"{datadir}/results_{suffix}.npy")
yy = (test_y*demandeur_std)+demandeur_mean

mean_sq_error = np.sqrt((xx[:, 0]-yy)**2).mean()
st.write(" Mean squared error: %f " % (mean_sq_error))
rel_error = 100*(xx[:, 0]-yy)/yy
st.write(" Relative Error: %1.3f ± %1.3f %% " %
         (rel_error.mean(), rel_error.std()))

fig, ax = plt.subplots(layout='constrained')
plt.plot([0, 60000], [0, 60000])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, r in enumerate(regions):
    plt.scatter(yy[test_x['REGION_NAME'] == i], xx[test_x['REGION_NAME'] == i],
                c=colors[i], label=r, alpha=0.5)
ax.set_xlabel("Expected")
ax.set_ylabel("Estimated")
ax.legend()
ax.set_title("Erreur en fonction de la région")
st.write(fig)

fig, ax = plt.subplots(layout='constrained')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
plt.plot([0, 60000], [0, 60000])
for i, c in enumerate(categories):
    plt.scatter(yy[test_x['REGISTRATION_CATEGORY_CODE'] == i], xx[test_x['REGISTRATION_CATEGORY_CODE'] == i],
                c=colors[i], label=c, alpha=0.5)
ax.set_xlabel("Expected")
ax.set_ylabel("Estimated")
ax.legend()
ax.set_title("Erreur en fonction de la catégorie")
st.write(fig)

fig, ax = plt.subplots(layout='constrained')
plt.scatter(yy, rel_error,
            c=colors[i], label=c, alpha=0.5)
ax.set_xlabel("Expected")
ax.set_title("Relative error (%)")
st.write(fig)

# Testing the model
st.write("## Test le modèle")
selectedRegion = st.selectbox("Sélectionner une région:", regions)
selectedMonth = st.selectbox("Sélectionner un mois:", [i for i in range(1,13)])
selectedCategory = st.selectbox("Sélectionner une catégorie:", categories)
selectedDemandeur = st.slider("Séléctionner un nombre de demandeurs:",min_value=0, max_value=70000, value=10000)

inputlabels = ['MONTH', 'REGION_NAME', 'REGISTRATION_CATEGORY_CODE', 'DEMANDEURS']
inputdata = [selectedMonth-1, regions.tolist().index(selectedRegion), categories.tolist().index(selectedCategory), selectedDemandeur]

inputDict = {k:np.array(v,dtype="int32")[np.newaxis,np.newaxis] for k,v in zip(inputlabels,inputdata)}

outDemandeur = (model.predict(inputDict)*demandeur_std)+demandeur_mean
st.metric(label="Estimated", value=int(outDemandeur), delta=int(outDemandeur-selectedDemandeur))