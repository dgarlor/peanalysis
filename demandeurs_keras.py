# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:07:49 2022

@author: dgarcialorenzo
"""


import matplotlib.pyplot as plt
from tensorflow.keras.layers import CategoryEncoding
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import IntegerLookup
import pandas as pd
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
BATCH = 8

idir = r"C:\cpp\peanalysis\data"
df = pd.read_pickle(f"{idir}/sagg_Demandeurs2012.2016.pkl")
df = df.sort_values(["REGISTRATION_CATEGORY_CODE", "REGION_NAME", "DATE"])
print(df.info())

regions = df.REGION_NAME.unique()
categories = df.REGISTRATION_CATEGORY_CODE.unique()


for c in ["MONTH"]:
    df[c] = pd.Categorical(df[c])

df = df.drop(columns=['DATE'])

# CONVERT TO categories
df.REGISTRATION_CATEGORY_CODE = df.REGISTRATION_CATEGORY_CODE.map(
    {c: i for i, c in enumerate(categories)})
df.REGION_NAME = df.REGION_NAME.map({c: i for i, c in enumerate(regions)})

numpycols = ["MONTH", "REGION_NAME",
             "REGISTRATION_CATEGORY_CODE", "DEMANDEURS", "EXPECTED"]

MONTH = 0
REGION_NAME = 1
REGISTRATION_CATEGORY_CODE = 2
DEMANDEURS = 3
EXPECTED = 4

all_ng = []
for r in range(len(regions)):
    for c in range(len(categories)):
        group = df.loc[(df.REGION_NAME == r) & (
            df.REGISTRATION_CATEGORY_CODE == c)]

        ng = group.to_numpy()
        # previous step
        # adding solution
        ng = np.append(ng[:-1, :], ng[1:, -1, np.newaxis], axis=1)
        all_ng.append(ng)

total = np.concatenate(all_ng, axis=0)
np.random.shuffle(total)

total[:, MONTH] -= 1  # to start with zero
N = total.shape[0]
Ntest = N // 5
Ntrain = N - Ntest
test_data = total[:Ntest]
train_data = total[Ntest:]


# Normalize demandeur values
demandeur_mean = np.mean(train_data, axis=0)[DEMANDEURS]
demandeur_std = np.std(train_data, axis=0)[DEMANDEURS]


def normalizeArray(arr, mean, std):
    return (arr.astype("float32")-mean)/std


def prepareData(arr):
    inputs = {}
    for i, nc in enumerate(numpycols[:-1]):
        inputs[nc] = arr[:, i]

    outputs = normalizeArray(arr[:, -1], demandeur_mean, demandeur_std)
    return (inputs, outputs)


test_x, test_y = prepareData(test_data)
train_x, train_y = prepareData(train_data)


print(
    "Using %d samples for training and %d for validation"
    % (len(train_data), len(test_data))
)


@tf.function
def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def encode_categorical_feature_embedings(feature, name, dataset, is_string):
    # TODO
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = IntegerLookup(output_mode="one_hot")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


month = layers.Input(shape=(1,), name="MONTH", dtype="int32")
region = layers.Input(shape=(1,), name="REGION_NAME", dtype="int32")
category = layers.Input(
    shape=(1,), name="REGISTRATION_CATEGORY_CODE", dtype="int32")
demandeurs = layers.Input(shape=(1,), name="DEMANDEURS", dtype="int32")

all_inputs = [month, region, category, demandeurs]

use_onehot = True
if use_onehot:
    m = CategoryEncoding(output_mode="one_hot", num_tokens=12)(month)
    r = CategoryEncoding(output_mode="one_hot",
                         num_tokens=regions.size)(region)
    c = CategoryEncoding(output_mode="one_hot",
                         num_tokens=categories.size)(category)
    d = Normalization(mean=demandeur_mean,
                      variance=demandeur_std**2)(demandeurs)
else:

    m = layers.Flatten()(layers.Embedding(input_dim=12, output_dim=3)(month))
    r = layers.Flatten()(layers.Embedding(input_dim=regions.size, output_dim=3)(region))
    c = layers.Flatten()(layers.Embedding(
        input_dim=categories.size, output_dim=3)(category))
    d = Normalization(mean=demandeur_mean,
                      variance=demandeur_std**2)(demandeurs)

# registration_encoded = layers.Embedding(input_dim= nvalues["REGISTRATION_CATEGORY_CODE"],
#                                         output_dim= 2)(registration)
# departement_encoded = layers.Embedding(input_dim= nvalues["DEPARTEMENT_NAME"],
#                                         output_dim= 10)(departement)
# age_encoded = layers.Embedding(input_dim= nvalues["AGE_GROUP_NAME"],
#                                         output_dim= 5)(age)
# month_encoded = layers.Embedding(       input_dim= nvalues["MONTH"],
#                                         output_dim= 5)(month)
# rome2_encoded = layers.Embedding(       input_dim= nvalues["ROME2"],
#                                         output_dim= 20)(rome2)

# gender_encoded = layers.Embedding(       input_dim= nvalues["GENDER_CODE"],
#                                         output_dim= 1)(gender)

all_features = layers.Concatenate()(
    [
        m, r, c, d
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Concatenate()([x, d])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Concatenate()([x, d])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.4)(x)
output = layers.Add()([layers.Dense(1)(x), d])
model = tf.keras.Model(all_inputs, output)
model.compile("adam", "mean_absolute_error", metrics=["accuracy"])
# mean_absolute_error?
model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
history = model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y))

model.save(".private/learned_model.keras")
import pickle

test_output = model.predict(test_x)
xx = (test_output*demandeur_std)+demandeur_mean


plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.scatter(xx, test_data[:, -1], s=5*test_data[:,
            REGISTRATION_CATEGORY_CODE], c=test_data[:, REGION_NAME], alpha=0.5)
plt.show()

plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.scatter(xx, test_data[:, -1], c=test_data[:,
            REGISTRATION_CATEGORY_CODE], alpha=0.5)
plt.show()
