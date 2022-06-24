from tensorflow.keras.layers import Normalization, CategoryEncoding
from tensorflow.keras import layers
import tensorflow as tf


def firstmodel(use_embeddings, regions, categories, demandeur_mean, demandeur_std):
    month = layers.Input(shape=(1,), name="MONTH", dtype="int32")
    region = layers.Input(shape=(1,), name="REGION_NAME", dtype="int32")
    category = layers.Input(
        shape=(1,), name="REGISTRATION_CATEGORY_CODE", dtype="int32")
    demandeurs = layers.Input(shape=(1,), name="DEMANDEURS", dtype="int32")

    all_inputs = [month, region, category, demandeurs]

    if not use_embeddings:
        m = CategoryEncoding(output_mode="one_hot", num_tokens=12)(month)
        r = CategoryEncoding(output_mode="one_hot",
                             num_tokens=regions.size)(region)
        c = CategoryEncoding(output_mode="one_hot",
                             num_tokens=categories.size)(category)
    else:
        m = layers.Flatten()(layers.Embedding(input_dim=12, output_dim=3)(month))
        r = layers.Flatten()(layers.Embedding(input_dim=regions.size, output_dim=3)(region))
        c = layers.Flatten()(layers.Embedding(
            input_dim=categories.size, output_dim=3)(category))

    d = Normalization(mean=demandeur_mean,
                      variance=demandeur_std**2)(demandeurs)
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
    return tf.keras.Model(all_inputs, output)
