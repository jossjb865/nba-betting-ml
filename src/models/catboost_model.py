"""Modelos CatBoost: clasificacion Moneyline y regresion de Totales.

Justificacion: las features tabulares del pipeline (diferenciales de forma,
fatiga, lesiones, contexto) tienen interacciones no lineales y variables
categoricas nativas (equipo local/visitante, mes, dia de semana, SeasonType)
que CatBoost maneja sin one-hot encoding mediante ordered target statistics,
lo que reduce leakage frente a encoding clasico.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

logger = logging.getLogger(__name__)


class CatBoostMoneyline:
    """Clasificador probabilistico: P(gana el equipo local)."""

    def __init__(self, params: Optional[Dict] = None,
                 categorical_features: Optional[List[str]] = None):
        self.params = params or {}
        self.categorical_features = categorical_features or []
        self.model: Optional[CatBoostClassifier] = None
        self.feature_cols: List[str] = []

    def _pool(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Pool:
        X = df[self.feature_cols].copy()
        for c in self.categorical_features:
            if c in X.columns:
                X[c] = X[c].astype("Int64").astype(str).fillna("NA")
        X = X.fillna(np.nan)
        cat_idx = [X.columns.get_loc(c) for c in self.categorical_features if c in X.columns]
        return Pool(X, label=target, cat_features=cat_idx)

    def fit(self, df: pd.DataFrame, feature_cols: List[str],
            eval_df: Optional[pd.DataFrame] = None) -> None:
        self.feature_cols = feature_cols
        train = df.dropna(subset=["Target_HomeWin"])
        y = train["Target_HomeWin"].astype(int)
        params = {"loss_function": "Logloss", "eval_metric": "Logloss",
                  "random_seed": 42, "verbose": False, **self.params}
        self.model = CatBoostClassifier(**params)
        eval_set = None
        if eval_df is not None:
            ev = eval_df.dropna(subset=["Target_HomeWin"])
            eval_set = self._pool(ev, ev["Target_HomeWin"].astype(int))
        self.model.fit(self._pool(train, y), eval_set=eval_set,
                       use_best_model=eval_set is not None)
        logger.info("CatBoost Moneyline entrenado con %d partidos", len(train))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(self._pool(df))[:, 1]

    def feature_importance(self) -> pd.Series:
        assert self.model is not None
        return pd.Series(self.model.get_feature_importance(),
                         index=self.feature_cols).sort_values(ascending=False)


class CatBoostTotals:
    """Regresor de puntos combinados (Over/Under).

    Se entrenan DOS regresores (puntos local y puntos visitante) para que el
    total predicho sea la suma; esto permite ademas alimentar la comparacion
    con la Poisson bivariada en la capa de fusion.
    """

    def __init__(self, params: Optional[Dict] = None,
                 categorical_features: Optional[List[str]] = None):
        self.params = {"loss_function": "RMSE", "eval_metric": "RMSE",
                       "random_seed": 42, "verbose": False, **(params or {})}
        self.categorical_features = categorical_features or []
        self.model_home: Optional[CatBoostRegressor] = None
        self.model_away: Optional[CatBoostRegressor] = None
        self.feature_cols: List[str] = []

    def _pool(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Pool:
        X = df[self.feature_cols].copy()
        for c in self.categorical_features:
            if c in X.columns:
                X[c] = X[c].astype("Int64").astype(str).fillna("NA")
        cat_idx = [X.columns.get_loc(c) for c in self.categorical_features if c in X.columns]
        return Pool(X, label=target, cat_features=cat_idx)

    def fit(self, df: pd.DataFrame, feature_cols: List[str],
            eval_df: Optional[pd.DataFrame] = None) -> None:
        self.feature_cols = feature_cols
        train = df.dropna(subset=["HomeTeamScore", "AwayTeamScore"])
        eval_set_h = eval_set_a = None
        if eval_df is not None:
            ev = eval_df.dropna(subset=["HomeTeamScore", "AwayTeamScore"])
            eval_set_h = self._pool(ev, ev["HomeTeamScore"])
            eval_set_a = self._pool(ev, ev["AwayTeamScore"])
        self.model_home = CatBoostRegressor(**self.params)
        self.model_away = CatBoostRegressor(**self.params)
        self.model_home.fit(self._pool(train, train["HomeTeamScore"]),
                            eval_set=eval_set_h, use_best_model=eval_set_h is not None)
        self.model_away.fit(self._pool(train, train["AwayTeamScore"]),
                            eval_set=eval_set_a, use_best_model=eval_set_a is not None)
        logger.info("CatBoost Totals entrenado con %d partidos", len(train))

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (puntos_local, puntos_visitante, total)."""
        assert self.model_home is not None and self.model_away is not None
        h = self.model_home.predict(self._pool(df))
        a = self.model_away.predict(self._pool(df))
        return h, a, h + a
