# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:07:49 2022

@author: dgarcialorenzo
"""


from demandeurs_keras_model import firstmodel
from demandeurs_keras_preprocessing import prepareData
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
BATCH = 8

idir = r"C:\cpp\peanalysis\data"
df = pd.read_pickle(f"{idir}/sagg_Demandeurs2012.2016.pkl")
regions = df.REGION_NAME.unique()
categories = df.REGISTRATION_CATEGORY_CODE.unique()
MONTH = 0
REGION_NAME = 1
REGISTRATION_CATEGORY_CODE = 2
DEMANDEURS = 3
EXPECTED = 4

(train_x, train_y), (test_x, test_y), demandeur_mean, demandeur_std = prepareData(df)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_y), len(test_y))
)

use_embeddings = True
model = firstmodel(use_embeddings, regions, categories,
                   demandeur_mean, demandeur_std)
suffix = "emb" if use_embeddings else "oneh"
model.compile("adam", "mean_absolute_error", metrics=["accuracy"])
# mean_absolute_error?
model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
history = model.fit(train_x, train_y, epochs=100,
                    validation_data=(test_x, test_y))

model.save(f"stream_data/learned_model_{suffix}.keras")
dfh = pd.DataFrame(history.history)
dfh.to_pickle(f"stream_data/learned_history_{suffix}.pkl")

test_output = model.predict(test_x)
xx = (test_output*demandeur_std)+demandeur_mean
yy = (test_y*demandeur_std)+demandeur_mean

plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.scatter(xx, yy, s=5*test_x['REGISTRATION_CATEGORY_CODE'],
            c=test_x['REGION_NAME'], alpha=0.5)
plt.show()

plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.scatter(xx, yy, c=test_x['REGISTRATION_CATEGORY_CODE'], alpha=0.5)
plt.show()
