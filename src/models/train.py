from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


@dataclass
class FittedModel:
    model: object
    model_name: str
    categorical: list[str]
    numeric: list[str]


def chronological_split(df: pd.DataFrame, train_fraction=0.70, validation_fraction=0.15):
    data = df.sort_values(["CreatedAt", "TicketID"]).reset_index(drop=True)
    n = len(data)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + validation_fraction))
    return data.iloc[:a].copy(), data.iloc[a:b].copy(), data.iloc[b:].copy()


def _preprocessor(categorical, numeric):
    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ])


def fit_model(model_name: str, train_df: pd.DataFrame, categorical: list[str], numeric: list[str], random_state=42) -> FittedModel:
    X = train_df[categorical + numeric].copy()
    y = train_df["SLABreached"].astype(int)
    for col in categorical:
        X[col] = X[col].astype("string")

    if model_name == "Logistic Regression":
        model = Pipeline([
            ("preprocess", _preprocessor(categorical, numeric)),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
        ])
        model.fit(X, y)
    elif model_name == "XGBoost":
        neg, pos = (y == 0).sum(), (y == 1).sum()
        model = Pipeline([
            ("preprocess", _preprocessor(categorical, numeric)),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, objective="binary:logistic", eval_metric="auc",
                scale_pos_weight=neg / max(pos, 1), random_state=random_state, n_jobs=-1,
            )),
        ])
        model.fit(X, y)
    elif model_name == "CatBoost":
        Xc = X.copy()
        for col in categorical:
            Xc[col] = Xc[col].fillna("__missing__").astype(str)
        med = Xc[numeric].median()
        Xc[numeric] = Xc[numeric].fillna(med)
        model = CatBoostClassifier(
            iterations=350, depth=5, learning_rate=0.03, loss_function="Logloss",
            eval_metric="AUC", auto_class_weights="Balanced", random_seed=random_state,
            verbose=False, allow_writing_files=False,
        )
        model.fit(Xc, y, cat_features=categorical)
        model._northbridge_numeric_medians = med.to_dict()
    else:
        raise ValueError(model_name)
    return FittedModel(model=model, model_name=model_name, categorical=categorical, numeric=numeric)


def predict_proba(fitted: FittedModel, df: pd.DataFrame) -> np.ndarray:
    X = df[fitted.categorical + fitted.numeric].copy()
    for col in fitted.categorical:
        X[col] = X[col].astype("string")
    if fitted.model_name == "CatBoost":
        for col in fitted.categorical:
            X[col] = X[col].fillna("__missing__").astype(str)
        med = getattr(fitted.model, "_northbridge_numeric_medians", {})
        for col in fitted.numeric:
            X[col] = X[col].fillna(med.get(col, X[col].median()))
    return fitted.model.predict_proba(X)[:, 1]
