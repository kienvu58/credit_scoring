import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn import model_selection
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from copy import copy as make_copy
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier


class MLAEFeature:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            subsample=0.8,
            silent=1,
            seed=50,
            reg_lambda=40,
            reg_alpha=10,
            objective="binary:logistic",
            n_estimators=1024,
            min_child_weight=15,
            max_depth=4,
            learning_rate=0.05,
            gamma=0.8,
            colsample_bytree=0.4,
            class_weight="d",
            verbose=2,
        )
        self.name = "MLAE Feature selection"

    def best_iter(self, X_train, y_train, X_test, y_test):
        print(
            "Selected metrics for evaluation: Logloss and AUC with default No. estimators = 1024 \n"
        )
        eval_set = [(X_train, y_train), (X_test, y_test)]
        eval_metric = ["logloss", "auc"]
        self.model.fit(
            X_train,
            y_train,
            eval_metric=eval_metric,
            eval_set=eval_set,
            verbose=True,
            early_stopping_rounds=20,
        )
        return self.model.best_iteration

    def feature_select(self, best_iteration, X_train, y_train, X_test, y_test):
        if "n_estimators" in self.model.get_params().keys():
            self.model.set_params(n_estimators=best_iteration)
        # Create selection model
        selection_model = make_copy(self.model)
        # Fit selection model
        self.model.fit(X_train, y_train)
        # Dump feature importance df
        feature_importance = pd.DataFrame(
            self.model.feature_importances_,
            columns=["gain_score"],
            index=X_train.columns,
        )
        feature_importance.to_excel("feature_importance.xlsx")
        # Feature selection loop
        thresholds = np.sort(self.model.feature_importances_)[
            np.nonzero(np.sort(self.model.feature_importances_))
        ][::-1]
        print(
            "Test model performance on original dataset with n_estimators = {} \n".format(
                self.model.get_params()["n_estimators"]
            )
        )
        for thresh in thresholds:
            selection = SelectFromModel(self.model, threshold=thresh, prefit=True)
            select_X_train = selection.transform(X_train)
            # Model defining
            selection_model.fit(select_X_train, y_train)
            # Model evaluation
            select_X_test = selection.transform(X_test)
            y_pred_select = selection_model.predict_proba(select_X_test)[:, 1]
            auc_select = metrics.roc_auc_score(y_test, y_pred_select)
            accuracy_select = metrics.accuracy_score(
                y_test, selection_model.predict(select_X_test)
            )
            print(
                "Thresh={:.9f}, n={}, Accuracy:  {:.2f}%, AUC: {:.2f}%".format(
                    thresh,
                    select_X_train.shape[1],
                    accuracy_select * 100.0,
                    auc_select * 100.0,
                )
            )
        pass


class MLAEBin:
    def __init__(self, max_depth):
        self.name = "MLAE Feature binning"
        self.tree_model = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=10, min_samples_split=10
        )

    def feature_bin(self, X, y):
        feature_name = input("Please input attribute name: ")
        self.tree_model.fit(X[feature_name].to_frame(), y)
        X["bin"] = self.tree_model.predict_proba(X[feature_name].to_frame())[:, 1]
        y_pred = self.tree_model.predict(X[feature_name].to_frame())
        fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        print(
            "With max depth = {}, AUC score is {:.2f}".format(
                self.tree_model.get_params()["max_depth"], roc_auc
            )
        )
        df = pd.concat(
            [
                X.groupby(["bin"][feature_name]).min(),
                X.groupby(["bin"][feature_name]).max(),
            ],
            axis=1,
        )
        print(df.head(10))
        return df


class MLAE(MLAEFeature, MLAEBin):
    def __init__(self):
        self.name = "MLAE"

    def help(self):
        print(
            """
        {} module for model optimization, including: \n 
        - Feature selection
        - Numerical attribute binning \n 
        - RandomCVSearch \n 
        - GridCVSearch 
        """.format(
                self.name
            )
        )
        pass
