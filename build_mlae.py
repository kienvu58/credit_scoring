import pandas as pd
import numpy as np
import time
import interest
import profile
import preprocess
import mlae
import joblib
import warnings


def data_prep(seed):
    profile = profile.Profile()
    interest = interest.Interest()
    preprocess = preprocess.Preprocessor()
    profile_raw = profile.get_profile()
    interest_raw, ids = interest.data_merge()
    data = preprocess.finalize_data(profile_raw, interest_raw)
    X, y, X_train, y_train, X_test, y_test = preprocess.split_data(
        data, seed=seed, re=False
    )
    return X, y, X_train, y_train, X_test, y_test, ids


def feature_extraction():

    return None


if __name__ == "__main__":
    print("*" * 100)
    print("*" * 100 + "\n")
    seed = 50
    X, y, X_train, y_train, X_test, y_test, ids = data_prep(seed)
    # Feature selection
    # mlae = mlae.MLAEFeature()
    # best_iter = mlae.best_iter(X_train, y_train, X_test, y_test)
    # mlae.feature_select(best_iter, X_train, y_train, X_test, y_test)
    # Numerical attribute binning
    mlae = mlae.MLAEBin(max_depth=2)
    bin = mlae.feature_bin(X, y)
    print("*" * 100)
    print("*" * 100 + "\n")
