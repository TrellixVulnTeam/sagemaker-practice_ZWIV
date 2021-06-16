"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    os.unlink(fn)

    logger.debug("Defining transformers.")
    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(df))
    X = df.copy()
    y = X.pop("rings")
    X_train, X_vt, y_train, y_vt = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_vt, y_vt, test_size=0.5, random_state=42
    )

    logger.info("Applying transforms.")
    X_train = preprocess.fit_transform(X_train)
    X_validation = preprocess.transform(X_validation)
    X_test = preprocess.transform(X_test)
    df_train = np.concatenate((y_train.to_numpy().reshape(-1, 1), X_train), axis=1)
    df_validation = np.concatenate((y_validation.to_numpy().reshape(-1, 1), X_validation), axis=1)
    df_test = np.concatenate((y_test.to_numpy().reshape(-1, 1), X_test), axis=1)

    logger.info("Writing out datasets to %s.", base_dir)
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/validation").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df_train).to_csv(
        f"{base_dir}/train/train.csv", header=False, index=False
    )
    pd.DataFrame(df_validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(df_test).to_csv(
        f"{base_dir}/test/test.csv", header=False, index=False
    )
