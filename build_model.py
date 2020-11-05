
import pandas as pd
import numpy as np
import time
import interest
import profile
import preprocess
import model
import joblib
import warnings


def preprocess_data(profile_link, level_list, seed):
    profile = profile.Profile()
    interest = interest.Interest()
    preprocess = preprocess.Preprocessor()
    profile_raw = profile.get_profile(profile_link)
    interest_raw, ids = interest.data_merge(level_list)
    data = preprocess.finalize_data(profile_raw, interest_raw)
    X, y, X_train, y_train, X_test, y_test = preprocess.split_data(
        data, seed=seed, re=False
    )
    return X, y, X_train, y_train, X_test, y_test


def build_model(
    X,
    y,
    X_train,
    y_train,
    X_test,
    y_test,
    seed,
    method,
):
    model = model.Model()
    evc_meta = model.model_ensemble(X, y, method=method)
    model.model_predict(evc_meta, X_train, y_train, X_test, y_test, seed=seed)
    model.cross_validate(evc_meta, X, y, seed)
    print("Start dumping Meta classifier...")
    joblib.dump(evc_meta, "meta_clf.pkl")
    print("Done dumping Meta classifier ! \n")
    return evc_meta


if __name__ == "__main__":
    print(
        "***************************************************************************************"
    )
    print(
        "***************************************************************************************"
    )
    seed = 50
    level_list = [
        {"level": "LV1", "link": "training_data/M_TRAINING_CLEAN_LV1.csv"},
        {"level": "LV2", "link": "training_data/M_TRAINING_CLEAN_LV2.csv"},
        {"level": "LV3", "link": "training_data/M_TRAINING_CLEAN_LV3.csv"},
        {"level": "LV4", "link": "training_data/M_TRAINING_CLEAN_LV4.csv"},
        {"level": "LV5", "link": "training_data/M_TRAINING_CLEAN_LV5.csv"},
    ]
    profile_link = "training_data/M_TRAINING_CLEAN_2_DEMO.csv"
    print(" \n With v represents VotingClassifier and s represents StackingClassifier.")
    method = input("Please specify preferred method (v or s): ")
    warnings.filterwarnings("ignore", category=FutureWarning)
    X, y, X_train, y_train, X_test, y_test = preprocess_data(
        profile_link, level_list, seed
    )
    evc_meta = build_model(
        X, y, X_train, y_train, X_test, y_test, seed=seed, method=method
    )
    print(
        "***************************************************************************************"
    )
    print(
        "***************************************************************************************"
    )
