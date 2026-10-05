"""Pipeline diario de prediccion (pregame).

Flujo (ejecutado por GitHub Actions cada dia de partidos):
1. Games by Date: partidos del dia con lineas pregame (PointSpread, OverUnder, MLs).
2. Betting Markets by GameID: mejores cuotas disponibles por sportsbook (opcional,
   mejora la seleccion de precio frente al consenso del Game).
3. InjuredPlayers + StartingLineupsByDate: estado de bajas actualizado.
4. Reconstruccion de features con el historico (sin usar nada post-tip-off).
5. Prediccion de los 4 modelos -> meta-matriz -> Fusion MLP calibrada.
6. Staking: EV + Kelly fraccionado sobre cuotas reales del feed.
7. Envio del informe a Telegram.

Uso:
  python -m src.pipelines.predict_daily            # hoy
  python -m src.pipelines.predict_daily --date 2026-01-15
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from ..features.builder import FeatureBuilder
from ..ingestion.ingest import Ingestor
from ..notify.telegram import TelegramNotifier
from ..sportsdataio.client import SportsDataIOClient
from ..staking.kelly import StakingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("predict")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def best_odds_from_markets(markets_df: pd.DataFrame) -> dict:
    """Extrae la MEJOR cuota disponible por mercado desde BettingMarketsByGameID."""
    out: dict = {}
    if markets_df.empty:
        return out
    full = markets_df[markets_df["BettingPeriodTypeID"] == 1]  # FullGame
    ml = full[full["BettingBetTypeID"] == 1]
    for side, key in (("Home", "home_ml"), ("Away", "away_ml")):
        rows = ml[ml["BettingOutcomeType"] == side].dropna(subset=["PayoutAmerican"])
        if not rows.empty:
            # mejor cuota = mayor PayoutDecimal
            best = rows.loc[rows["PayoutDecimal"].astype(float).idxmax()]
            out[key] = float(best["PayoutAmerican"])
            out[key + "_book"] = best.get("SportsbookName")
    tot = full[full["BettingBetTypeID"] == 3]
    over = tot[tot["BettingOutcomeType"] == "Over"].dropna(subset=["PayoutAmerican"])
    under = tot[tot["BettingOutcomeType"] == "Under"].dropna(subset=["PayoutAmerican"])
    if not over.empty and not under.empty:
        bo = over.loc[over["PayoutDecimal"].astype(float).idxmax()]
        bu = under.loc[under["PayoutDecimal"].astype(float).idxmax()]
        out["total_line"] = float(bo["Value"]) if bo.get("Value") is not None else None
        out["over_odds"] = float(bo["PayoutAmerican"])
        out["under_odds"] = float(bu["PayoutAmerican"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (por defecto: hoy)")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    target_date = (datetime.strptime(args.date, "%Y-%m-%d").date()
                   if args.date else date.today())

    artifacts = Path(cfg["paths"]["artifacts"])
    required = ["catboost_moneyline.pkl", "catboost_totals.pkl", "bivariate_poisson.pkl",
                "lstm_momentum.pt", "fusion_mlp.pkl", "sequence_builder.pkl",
                "feature_config.json"]
    missing = [f for f in required if not (artifacts / f).exists()]
    if missing:
        raise RuntimeError(
            f"Faltan artefactos {missing}. Ejecuta primero el pipeline de entrenamiento "
            "(workflow weekly_retrain.yml o `python -m src.pipelines.train`).")

    import json
    with open(artifacts / "feature_config.json") as f:
        fcfg = json.load(f)
    with open(artifacts / "catboost_moneyline.pkl", "rb") as f:
        cb_ml = pickle.load(f)
    with open(artifacts / "catboost_totals.pkl", "rb") as f:
        cb_tot = pickle.load(f)
    with open(artifacts / "bivariate_poisson.pkl", "rb") as f:
        bp = pickle.load(f)
    with open(artifacts / "fusion_mlp.pkl", "rb") as f:
        fusion = pickle.load(f)
    with open(artifacts / "sequence_builder.pkl", "rb") as f:
        seq_builder = pickle.load(f)
    from ..models.lstm_momentum import LSTMMomentum
    lstm = LSTMMomentum()
    lstm.load(str(artifacts / "lstm_momentum.pt"), n_feats=len(seq_builder.feature_cols))

    # ---------------- Datos del dia (pregame) ---------------- #
    client = SportsDataIOClient(base_url=cfg["api"]["base_url"],
                                timeout=cfg["api"]["timeout_seconds"],
                                max_retries=cfg["api"]["max_retries"],
                                min_call_interval=cfg["api"]["min_call_interval"])
    ing = Ingestor(client, cfg["paths"]["raw_data"])

    games_today = ing.fetch_games_by_date(target_date)
    games_today = games_today[~games_today["Status"].isin(["Final", "F/OT"])]
    if games_today.empty:
        logger.info("No hay partidos pendientes para %s", target_date)
        if cfg["telegram"]["send_no_bet_summary"]:
            TelegramNotifier(parse_mode=cfg["telegram"]["parse_mode"]).send(
                f"*NBA Model - {target_date.isoformat()}*\n\nSin partidos programados hoy.")
        return

    injuries = ing.fetch_injuries(target_date)
    ing.fetch_starting_lineups(target_date)

    # Historico actualizado (para features rolling) + registro del dia actual
    games_hist = ing.load_all_games()
    team_games_hist = ing.load_all_team_games()
    player_games_hist = ing.load_all_player_games()
    player_season = None
    season = int(cfg["season"]["current"])
    ps_path = Path(cfg["paths"]["raw_data"]) / f"player_season_{season}.parquet"
    if ps_path.exists():
        player_season = pd.read_parquet(ps_path)

    # Combinar calendario historico con los partidos de hoy (targets = NaN)
    games_all = pd.concat([games_hist, games_today], ignore_index=True) \
                  .drop_duplicates(subset=["GameID"], keep="last")
    games_all["Day"] = pd.to_datetime(games_all["Day"])

    fb = FeatureBuilder(cfg["features"]["rolling_windows"],
                        cfg["features"]["min_games_history"])
    feats = fb.build(games_all, team_games_hist, player_games=player_games_hist,
                     injuries_snapshots=injuries, player_season=player_season)
    today_ids = set(games_today["GameID"])
    today_feats = feats[feats["GameID"].isin(today_ids)].copy()
    if today_feats.empty:
        raise RuntimeError("No se pudieron construir features para los partidos de hoy "
                           "(historico insuficiente).")

    # ---------------- Predicciones de modelos base ---------------- #
    today_feats["p_cb_ml"] = cb_ml.predict_proba(today_feats)
    _, _, today_feats["cb_total"] = cb_tot.predict(today_feats)

    tg = team_games_hist.copy()
    pts = tg.pivot_table(index="GameID", columns="HomeOrAway", values="Points", aggfunc="first")
    tg["OppPoints"] = tg.apply(
        lambda r: pts.loc[r["GameID"]].drop(r["HomeOrAway"], errors="ignore").iloc[0]
        if r["GameID"] in pts.index and len(pts.loc[r["GameID"]].dropna()) > 1 else np.nan, axis=1)
    tg["Day"] = pd.to_datetime(tg["Day"])
    Xh, Xa, ids = seq_builder.build(tg, games_all)
    mask = np.isin(ids, list(today_ids))
    if mask.sum():
        p_lstm = lstm.predict_proba(Xh[mask], Xa[mask])
        t_lstm = lstm.predict_total(Xh[mask], Xa[mask])
        map_lstm = dict(zip(ids[mask], zip(p_lstm, t_lstm)))
        today_feats["p_lstm_ml"] = today_feats["GameID"].map(
            lambda g: map_lstm.get(g, (np.nan, np.nan))[0])
        today_feats["total_lstm"] = today_feats["GameID"].map(
            lambda g: map_lstm.get(g, (np.nan, np.nan))[1])
    today_feats["p_lstm_ml"] = today_feats["p_lstm_ml"].fillna(today_feats["p_cb_ml"])
    today_feats["total_lstm"] = today_feats["total_lstm"].fillna(today_feats["cb_total"])

    bp_rows = today_feats.apply(
        lambda r: bp.predict_markets(int(r["HomeTeamID"]), int(r["AwayTeamID"]),
                                     r.get("OverUnder")), axis=1)
    today_feats["p_bp_ml"] = [r["P_home_win"] for r in bp_rows]
    today_feats["total_bp"] = [r["ExpectedTotal"] for r in bp_rows]
    today_feats["p_bp_over"] = [r.get("P_over", np.nan) for r in bp_rows]

    # ---------------- Fusion ---------------- #
    meta = pd.DataFrame({
        "p_cb_ml": today_feats["p_cb_ml"],
        "p_lstm_ml": today_feats["p_lstm_ml"],
        "p_bp_ml": today_feats["p_bp_ml"],
        "total_lstm": today_feats["total_lstm"],
        "total_bp": today_feats["total_bp"],
        "market_total_line": today_feats["OverUnder"],
        "diff_total_vs_line": (today_feats["total_lstm"] + today_feats["total_bp"]) / 2
                              - today_feats["OverUnder"],
    })
    p_home = fusion.predict_proba(meta)
    total_pred = fusion.predict_total(meta)
    resid_std = float(fcfg.get("residual_total_std", 12.0))
    # P(Over) combinando la estimacion fusionada con la Poisson bivariada
    p_over_norm = 1 - norm.cdf((today_feats["OverUnder"].to_numpy(float) - total_pred) / resid_std)
    p_over_bp = today_feats["p_bp_over"].to_numpy(float)
    p_over = np.where(np.isnan(p_over_bp), p_over_norm, 0.5 * p_over_norm + 0.5 * p_over_bp)

    # ---------------- Staking con cuotas reales ---------------- #
    staking = StakingEngine(
        bankroll_units=cfg["staking"]["bankroll_units"],
        kelly_fraction=cfg["staking"]["kelly_fraction"],
        max_stake_pct=cfg["staking"]["max_stake_pct"],
        min_edge=cfg["staking"]["min_edge"],
        min_ev=cfg["staking"]["min_ev"])

    picks, all_games = [], []
    for i, (_, row) in enumerate(today_feats.iterrows()):
        label = f"{row['AwayTeam']} @ {row['HomeTeam']}"
        # Prioridad: mejor cuota del feed de mercados; fallback: consenso del Game
        try:
            mk = ing.fetch_betting_markets(int(row["GameID"]))
            best = best_odds_from_markets(mk)
        except Exception as exc:  # si el feed de odds falla, se usa el consenso
            logger.warning("Odds de mercado no disponibles para %s: %s", row["GameID"], exc)
            best = {}
        home_ml = best.get("home_ml", row.get("HomeTeamMoneyLine"))
        away_ml = best.get("away_ml", row.get("AwayTeamMoneyLine"))
        line = best.get("total_line", row.get("OverUnder"))
        over_odds = best.get("over_odds", row.get("OverPayout"))
        under_odds = best.get("under_odds", row.get("UnderPayout"))

        decisions = staking.evaluate_game(label, float(p_home[i]), float(p_over[i]),
                                          home_ml, away_ml, line, over_odds, under_odds)
        for d in decisions:
            if d.bet:
                picks.append({"market": d.market, "selection": d.selection,
                              "p_model": d.p_model, "edge": d.edge, "ev": d.ev,
                              "american_odds": d.american_odds, "stake_units": d.stake_units})
        all_games.append({"away": row["AwayTeam"], "home": row["HomeTeam"],
                          "p_home": float(p_home[i]), "total": float(total_pred[i]),
                          "line": line})

    # ---------------- Telegram ---------------- #
    notifier = TelegramNotifier(parse_mode=cfg["telegram"]["parse_mode"],
                                max_len=cfg["telegram"]["max_message_length"])
    msg = notifier.format_predictions(target_date.isoformat(), picks, all_games)
    ok = notifier.send(msg)
    logger.info("Telegram enviado=%s | value bets=%d", ok, len(picks))

    out = Path(cfg["paths"]["reports"]); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(picks).to_csv(out / f"picks_{target_date.isoformat()}.csv", index=False)
    pd.DataFrame(all_games).to_csv(out / f"games_{target_date.isoformat()}.csv", index=False)


if __name__ == "__main__":
    main()
