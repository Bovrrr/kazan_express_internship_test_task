#!/usr/bin/env python3


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ruBERTEmbeddings import ruBERTEmbeddings
from constants import *

data_path = "../../data/"


# Загружаю данные
train_data = pd.read_parquet(data_path + "train.parquet")
test_data = pd.read_parquet(data_path + "test.parquet")


# спличу фулл трейн на Трейн и валидацию
tr_data, te_data = train_test_split(
    train_data,
    test_size=0.3,
    stratify=train_data["category_id"],
    random_state=RANDOM_STATE,
)
del train_data

# выделяю Таргет
tr_y = tr_data["category_id"]
pd.DataFrame(tr_y).to_parquet(
    "tr_y.parquet",
    engine="fastparquet",
    index=True,
)
del tr_y

te_y = te_data["category_id"]
pd.DataFrame(te_y).to_parquet(
    "tr_y.parquet",
    engine="fastparquet",
    index=True,
)
del te_y


# убираю Айди товара и Таргет из данных
bad_feats = ["short_description", "name_value_characteristics"]

tr_data = tr_data[tr_data.columns.tolist()[1:-1]].drop(
    bad_feats,
    axis=1,
)
te_data = te_data[te_data.columns.tolist()[1:-1]].drop(
    bad_feats,
    axis=1,
)
test_data.drop(
    bad_feats,
    axis=1,
    inplace=True,
)

# проверяю то, что в обеих частях одинаковое кол-во классов
assert tr_y.nunique() == te_y.nunique()


# создаю инстанс препроцессора текстовой фичи
title_idx = tr_data.columns.tolist().index("title")
reBert_trans = ruBERTEmbeddings(title_idx, "cointegrated/rubert-tiny").fit(tr_data)

tr_data = reBert_trans.transform(tr_data)
tr_data.columns = list(map(str, tr_data.columns))
tr_data.to_parquet(
    "tr_data.parquet",
    engine="fastparquet",
    index=True,
)
del tr_data

te_data = reBert_trans.transform(te_data)
te_data.columns = list(map(str, te_data.columns))
te_data.to_parquet(
    "te_data.parquet",
    engine="fastparquet",
    index=True,
)
del te_data


test_data_idx = test_data["id"]
test_data.drop(["id"], axis=1, inplace=True)
test_data = reBert_trans.transform(test_data)
test_data.columns = list(map(str, test_data.columns))
test_data.to_parquet(
    "test_data.parquet",
    engine="fastparquet",
    index=True,
)
del test_data
