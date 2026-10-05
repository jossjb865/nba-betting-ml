"""Pipeline de entrenamiento completo.

Pasos:
1. Backfill historico desde SportsDataIO (Games, TeamGame, PlayerGame por fecha).
2. Construccion de la matriz de features y secuencias LSTM.
3. Validacion walk-forward:
   - Genera predicciones OUT-OF-FOLD de los 3 modelos base.
   - Entrena la Fusion MLP sobre esas predicciones OOF (stacking sin leakage).
4. Entrena los modelos finales con TODO el historico disponible.
5. Persiste artefactos en models/artifacts/ y genera informe en reports/.

Uso:
  python -m src.pipelines.train --seasons 2023 2024 2025 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..features.builder import CATEGORICAL_FEATURES, FeatureBuilder, tabular_feature_columns
from ..features.sequences import SequenceBuilder
from ..ingestion.ingest import Ingestor
from ..models.bivariate_poisson import BivariatePoissonModel
from ..models.catboost_model import CatBoostMoneyline, CatBoostTotals
from ..models.fusion_mlp import FusionConfig, FusionMLP
from ..models.lstm_momentum import LSTMConfig, LSTMMomentum
from ..sportsdataio.client import SportsDataIOClient
from ..validation.walk_forward import (calibration_table, classification_metrics,
                                       regression_metrics, simulate_roi,
                                       walk_forward_splits)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seasons = args.seasons or cfg["season"]["training_seasons"]
    season_types = cfg["season"]["season_types"]

    artifacts = Path(cfg["paths"]["artifacts"]); artifacts.mkdir(parents=True, exist_ok=True)
    reports = Path(cfg["paths"]["reports"]); reports.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Ingestion ---------------- #
    client = SportsDataIOClient(base_url=cfg["api"]["base_url"],
                                timeout=cfg["api"]["timeout_seconds"],
                                max_retries=cfg["api"]["max_retries"],
                                min_call_interval=cfg["api"]["min_call_interval"])
    ing = Ingestor(client, cfg["paths"]["raw_data"])
    for s in seasons:
        ing.backfill_season(s, season_types)
        ing.fetch_player_season_stats(s)

    games = ing.load_all_games()
    team_games = ing.load_all_team_games()
    player_games = ing.load_all_player_games()
    games["Day"] = pd.to_datetime(games["Day"])
    logger.info("Historico cargado: %d partidos, %d lineas de equipo", len(games), len(team_games))

    # ---------------- 2. Features ---------------- #
    fb = FeatureBuilder(cfg["features"]["rolling_windows"],
                        cfg["features"]["min_games_history"])
    feats = fb.build(games, team_games, player_games=player_games)
    feats = feats.dropna(subset=["Target_HomeWin", "Target_TotalPoints"]).reset_index(drop=True)
    feats.to_parquet(Path(cfg["paths"]["processed_data"]) / "features_train.parquet", index=False)

    feat_cols = [c for c in tabular_feature_columns(feats) if c in feats.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feats.columns]

    # Enriquecer team_games para secuencias (margen, oponente)
    tg = team_games.copy()
    pts = tg.pivot_table(index="GameID", columns="HomeOrAway", values="Points", aggfunc="first")
    tg["OppPoints"] = tg.apply(
        lambda r: pts.loc[r["GameID"]].drop(r["HomeOrAway"], errors="ignore").iloc[0]
        if r["GameID"] in pts.index and len(pts.loc[r["GameID"]].dropna()) > 1 else np.nan, axis=1)
    tg["Day"] = pd.to_datetime(tg["Day"])

    # ---------------- 3. Walk-forward + stacking OOF ---------------- #
    oof = feats[["GameID", "Day"]].copy()
    oof["p_cb_ml"] = np.nan
    oof["cb_total"] = np.nan
    oof["p_bp_ml"] = np.nan
    oof["total_bp"] = np.nan
    oof["p_lstm_ml"] = np.nan
    oof["total_lstm"] = np.nan

    seq_builder = SequenceBuilder(cfg["features"]["lstm_sequence_length"])
    seq_builder.fit_normalization(tg)

    metrics_folds = []
    for k, df_sorted, tr, te in walk_forward_splits(
            feats, n_splits=cfg["validation"]["n_splits"],
            min_train=cfg["validation"]["min_train_games"]):
        train_df, test_df = df_sorted.iloc[tr], df_sorted.iloc[te]
        gids_test = set(test_df["GameID"])

        # CatBoost
        cb_ml = CatBoostMoneyline(cfg["models"]["catboost"]["moneyline"], cat_cols)
        cb_ml.fit(train_df, feat_cols)
        cb_tot = CatBoostTotals(cfg["models"]["catboost"]["totals"], cat_cols)
        cb_tot.fit(train_df, feat_cols)
        oof.loc[oof["GameID"].isin(gids_test), "p_cb_ml"] = cb_ml.predict_proba(test_df)
        _, _, tot_cb = cb_tot.predict(test_df)
        oof.loc[oof["GameID"].isin(gids_test), "cb_total"] = tot_cb

        # Poisson bivariada (solo marcadores de train)
        bp = BivariatePoissonModel(tuple(cfg["models"]["bivariate_poisson"]["lambda3_bounds"]),
                                   cfg["models"]["bivariate_poisson"]["shrinkage"])
        bp.fit(train_df.dropna(subset=["HomeTeamScore"]))
        bp_preds = test_df.apply(
            lambda r: bp.predict_markets(int(r["HomeTeamID"]), int(r["AwayTeamID"]),
                                         r.get("OverUnder")), axis=1)
        oof.loc[oof["GameID"].isin(gids_test), "p_bp_ml"] = [p["P_home_win"] for p in bp_preds]
        oof.loc[oof["GameID"].isin(gids_test), "total_bp"] = [p["ExpectedTotal"] for p in bp_preds]

        # LSTM momentum
        Xh, Xa, ids = seq_builder.build(tg, df_sorted)
        mask_tr = np.isin(ids, set(train_df["GameID"]))
        mask_te = np.isin(ids, gids_test)
        if mask_te.sum() > 0 and mask_tr.sum() > 100:
            id2target = df_sorted.set_index("GameID")
            lstm = LSTMMomentum(LSTMConfig(**cfg["models"]["lstm_momentum"]))
            lstm.fit(Xh[mask_tr], Xa[mask_tr],
                     id2target.loc[ids[mask_tr], "Target_HomeWin"].to_numpy(float),
                     id2target.loc[ids[mask_tr], "Target_TotalPoints"].to_numpy(float))
            p_te = lstm.predict_proba(Xh[mask_te], Xa[mask_te])
            t_te = lstm.predict_total(Xh[mask_te], Xa[mask_te])
            oof_idx = oof.index[oof["GameID"].isin(ids[mask_te])]
            oof.loc[oof_idx, "p_lstm_ml"] = p_te
            oof.loc[oof_idx, "total_lstm"] = t_te

        # Metricas del fold (CatBoost + Poisson como referencia)
        y_true = test_df["Target_HomeWin"].to_numpy(float)
        m = classification_metrics(y_true, oof.loc[oof["GameID"].isin(gids_test), "p_cb_ml"].to_numpy(float))
        m.update(regression_metrics(test_df["Target_TotalPoints"].to_numpy(float), tot_cb))
        m["fold"] = k
        metrics_folds.append(m)
        logger.info("Fold %d: %s", k, m)

    # ---------------- 4. Fusion MLP sobre OOF ---------------- #
    oof_full = oof.merge(feats[["GameID", "OverUnder", "Target_HomeWin", "Target_TotalPoints",
                                "HomeTeamMoneyLine", "AwayTeamMoneyLine",
                                "OverPayout", "UnderPayout"]], on="GameID")
    oof_full["p_lstm_ml"] = oof_full["p_lstm_ml"].fillna(oof_full["p_cb_ml"])
    oof_full["total_lstm"] = oof_full["total_lstm"].fillna(oof_full["cb_total"])
    meta = pd.DataFrame({
        "p_cb_ml": oof_full["p_cb_ml"],
        "p_lstm_ml": oof_full["p_lstm_ml"],
        "p_bp_ml": oof_full["p_bp_ml"],
        "total_lstm": oof_full["total_lstm"],
        "total_bp": oof_full["total_bp"],
        "market_total_line": oof_full["OverUnder"],
        "diff_total_vs_line": (oof_full["total_lstm"] + oof_full["total_bp"]) / 2
                              - oof_full["OverUnder"],
    })
    valid = meta.dropna().index
    fusion = FusionMLP(FusionConfig(**cfg["models"]["fusion_mlp"]))
    fusion.fit(meta.loc[valid], oof_full.loc[valid, "Target_HomeWin"].to_numpy(float),
               oof_full.loc[valid, "Target_TotalPoints"].to_numpy(float))

    # ---------------- 5. Evaluacion final de negocio (holdout OOF) ---------------- #
    p_final = fusion.predict_proba(meta.loc[valid])
    t_final = fusion.predict_total(meta.loc[valid])
    holdout = oof_full.loc[valid].copy()
    holdout["p_final"] = p_final
    # Probabilidad de Over a la linea de mercado: aproximacion normal sobre el total fusionado
    resid_std = float(np.nanstd(holdout["Target_TotalPoints"] - t_final)) or 12.0
    from scipy.stats import norm
    p_over = 1 - norm.cdf((holdout["OverUnder"] - t_final) / resid_std)
    biz = simulate_roi(holdout, p_final, p_over.to_numpy(),
                       kelly_fraction=cfg["staking"]["kelly_fraction"],
                       max_stake_pct=cfg["staking"]["max_stake_pct"],
                       min_edge=cfg["staking"]["min_edge"])
    final_cls = classification_metrics(holdout["Target_HomeWin"].to_numpy(float), p_final)
    final_reg = regression_metrics(holdout["Target_TotalPoints"].to_numpy(float), t_final)
    calib = calibration_table(holdout["Target_HomeWin"].to_numpy(float), p_final)

    # ---------------- 6. Re-entrenar con todo el historico y persistir ---------------- #
    cb_ml_final = CatBoostMoneyline(cfg["models"]["catboost"]["moneyline"], cat_cols)
    cb_ml_final.fit(feats, feat_cols)
    cb_tot_final = CatBoostTotals(cfg["models"]["catboost"]["totals"], cat_cols)
    cb_tot_final.fit(feats, feat_cols)
    bp_final = BivariatePoissonModel(tuple(cfg["models"]["bivariate_poisson"]["lambda3_bounds"]),
                                     cfg["models"]["bivariate_poisson"]["shrinkage"])
    bp_final.fit(feats.dropna(subset=["HomeTeamScore"]))
    lstm_final = LSTMMomentum(LSTMConfig(**cfg["models"]["lstm_momentum"]))
    Xh, Xa, ids = seq_builder.build(tg, feats)
    id2t = feats.set_index("GameID")
    lstm_final.fit(Xh, Xa, id2t.loc[ids, "Target_HomeWin"].to_numpy(float),
                   id2t.loc[ids, "Target_TotalPoints"].to_numpy(float))

    with open(artifacts / "catboost_moneyline.pkl", "wb") as f:
        pickle.dump(cb_ml_final, f)
    with open(artifacts / "catboost_totals.pkl", "wb") as f:
        pickle.dump(cb_tot_final, f)
    with open(artifacts / "bivariate_poisson.pkl", "wb") as f:
        pickle.dump(bp_final, f)
    lstm_final.save(str(artifacts / "lstm_momentum.pt"))
    with open(artifacts / "fusion_mlp.pkl", "wb") as f:
        pickle.dump(fusion, f)
    with open(artifacts / "sequence_builder.pkl", "wb") as f:
        pickle.dump(seq_builder, f)
    with open(artifacts / "feature_config.json", "w") as f:
        json.dump({"feature_cols": feat_cols, "categorical": cat_cols,
                   "residual_total_std": resid_std}, f, indent=2)

    # ---------------- 7. Informe ---------------- #
    report = {
        "folds": metrics_folds,
        "final_oof_classification": final_cls,
        "final_oof_regression": final_reg,
        "business_simulation_quarter_kelly": biz,
        "calibration": calib.to_dict(orient="records"),
        "feature_importance_top15": cb_ml_final.feature_importance().head(15).to_dict(),
        "n_games_trained": int(len(feats)),
        "seasons": seasons,
    }
    with open(reports / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Entrenamiento completado. ROI simulado: %.2f%% | LogLoss: %.4f",
                100 * biz.get("roi", float("nan")), final_cls["logloss"])


if __name__ == "__main__":
    main()
