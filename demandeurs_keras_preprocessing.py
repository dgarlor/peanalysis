# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:07:49 2022

@author: dgarcialorenzo
"""
import numpy as np


def prepareData(df):

    df = df.sort_values(["REGISTRATION_CATEGORY_CODE", "REGION_NAME", "DATE"])

    regions = df.REGION_NAME.unique()
    categories = df.REGISTRATION_CATEGORY_CODE.unique()
    df = df.drop(columns=['DATE'])

    # CONVERT TO categories
    df.REGISTRATION_CATEGORY_CODE = df.REGISTRATION_CATEGORY_CODE.map(
        {c: i for i, c in enumerate(categories)})
    df.REGION_NAME = df.REGION_NAME.map({c: i for i, c in enumerate(regions)})

    numpycols = ["MONTH", "REGION_NAME",
                 "REGISTRATION_CATEGORY_CODE", "DEMANDEURS", "EXPECTED"]

    MONTH = 0
    DEMANDEURS = 3

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

    return (train_x, train_y), (test_x, test_y), demandeur_mean, demandeur_std
