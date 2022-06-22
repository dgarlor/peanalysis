# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:40:20 2022

@author: dgarcialorenzo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:07:49 2022

@author: dgarcialorenzo
"""


import pandas as pd
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
BATCH = 8

idir = r"C:\cpp\peanalysis\data\demandeEmploi_dfs_old"
df = pd.read_pickle(f"{idir}/compilation_months.pkl")
print(df.info())

df['ROME2'] = df.ROME_PROFESSION_CARD_CODE.replace(r'\w{2}$','',regex=True)
df.MONTH = df.MONTH.map({1:"JAN",2:"FEV",3:"MAR",4:"APR",5:"MAY",6:"JUIN",
              7:"JUIL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"})
df.GENDER_CODE = df.GENDER_CODE.astype(int) -1

for c in ["REGISTRATION_CATEGORY_CODE","ROME2","DEPARTEMENT_NAME","AGE_GROUP_NAME","MONTH"]:
    df[c] = pd.Categorical(df[c])

df = df.drop(columns=['REGION_CODE', 'ROME_PROFESSION_CARD_NAME', 'REGION_NAME',
       'QUALIFICATION_NAME', 'ROME_PROFESSION_CARD_CODE',
       'DEPARTEMENT_CODE', 'GENDER_NAME', 'IDENT',
       'AGE_GROUP_CODE', '_id', 'QUALIFICATION_CODE', 'YEAR'])


val_dataframe = df.sample(frac=0.2, random_state=1337)
train_dataframe = df.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)



def dataframe_to_dataset(dataframe, shuffle=True, categoricals=[]):
    dataframe = dataframe.copy()
    labels = pd.get_dummies(dataframe.pop("MONTHS"))
    
    dd ={}
    for k,v in dict(dataframe).items():
        if k in categoricals:
            dd[k] = np.reshape(np.array(pd.Categorical(v).codes),(v.shape[0],1)) 
        else:
            dd[k] = np.reshape(np.array(v),(v.shape[0],1))

    ds = tf.data.Dataset.from_tensor_slices((dd, labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

categ = ['REGISTRATION_CATEGORY_CODE', 'DEPARTEMENT_NAME', 'AGE_GROUP_NAME', 'MONTH', 'ROME2','GENDER_CODE']
nvalues = {c:len(df[c].unique()) for c in df}
train_ds = dataframe_to_dataset(train_dataframe, categoricals = categ)
val_ds = dataframe_to_dataset(val_dataframe, shuffle=False, categoricals = categ)


for x, y in val_ds.take(1):
    print("Input:", x)
    print("Target:", y)


from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup

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
    #TODO
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


registration = layers.Input(shape=(1,),name="REGISTRATION_CATEGORY_CODE",dtype="int32")
departement = layers.Input(shape=(1,),name="DEPARTEMENT_NAME",dtype="int32")
age = layers.Input(shape=(1,),name="AGE_GROUP_NAME",dtype="int32")
month = layers.Input(shape=(1,),name="MONTH",dtype="int32")
rome2 = layers.Input(shape=(1,),name="ROME2",dtype="int32")

gender= layers.Input(shape=(1,),name="GENDER_CODE",dtype="int32")

all_inputs = [
    registration, departement, age, month, rome2, gender
    ]

# registration_encoded = encode_categorical_feature(registration, "REGISTRATION_CATEGORY_CODE", train_ds, True)
# departement_encoded = encode_categorical_feature(departement, "DEPARTEMENT_NAME", train_ds, True)
# age_encoded = encode_categorical_feature(age, "AGE_GROUP_NAME", train_ds, True)
# month_encoded = encode_categorical_feature(month, "MONTH", train_ds, True)
# rome2_encoded = encode_categorical_feature(rome2, "ROME2", train_ds, True)
# gender_encoded = encode_numerical_feature(gender, "GENDER_CODE", train_ds)

registration_encoded = layers.Embedding(input_dim= nvalues["REGISTRATION_CATEGORY_CODE"],
                                        output_dim= 2)(registration)
departement_encoded = layers.Embedding(input_dim= nvalues["DEPARTEMENT_NAME"],
                                        output_dim= 10)(departement)
age_encoded = layers.Embedding(input_dim= nvalues["AGE_GROUP_NAME"],
                                        output_dim= 5)(age)
month_encoded = layers.Embedding(       input_dim= nvalues["MONTH"],
                                        output_dim= 5)(month)
rome2_encoded = layers.Embedding(       input_dim= nvalues["ROME2"],
                                        output_dim= 20)(rome2)

gender_encoded = layers.Embedding(       input_dim= nvalues["GENDER_CODE"],
                                        output_dim= 1)(gender)

all_features = layers.Concatenate()(
    [
        registration_encoded,
        departement_encoded,
        age_encoded,
        month_encoded,
        rome2_encoded,
        gender_encoded
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Reshape((128,))(x)
output = layers.Dense(nvalues["MONTHS"],activation="softmax")(x)
model = tf.keras.Model(all_inputs, output)
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# mean_absolute_error?
model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
train_ds = train_ds.batch(BATCH)
val_ds = val_ds.batch(BATCH)
model.fit(train_ds, epochs=50, validation_data=val_ds)
