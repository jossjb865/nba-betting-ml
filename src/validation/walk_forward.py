"""Pipeline de validacion: walk-forward temporal estricto.

Regla anti-leakage: los splits se hacen por FECHA. Ningun partido del conjunto
de test precede cronologicamente a uno de train. Las medias moviles y rachas ya
se calculan con shift(1) en la capa de features.

Metricas:
- Moneyline (clasificacion): LogLoss, Brier Score, Accuracy, AUC, curva de calibracion.
- Totales (regresion): MAE, RMSE.
- Capa de negocio: ROI simulado con staking Quarter-Kelly sobre cuotas reales
  historicas (moneylines y totales del propio feed), y CLV (Closing Line Value)
  cuando se dispone de linea de cierre.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ..staking.kelly import american_to_decimal, devig_two_way, kelly_fraction_full

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    metrics: Dict[str, float]


def walk_forward_splits(df: pd.DataFrame, date_col: str = "Day",
                        n_splits: int = 6, min_train: int = 400):
    """Genera pares (train_idx, test_idx) expandiendo la ventana de train."""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    fold_sizes = (n - min_train) // n_splits
    if fold_sizes <= 0:
        raise ValueError(f"Datos insuficientes para {n_splits} splits con min_train={min_train}")
    for k in range(n_splits):
        test_start = min_train + k * fold_sizes
        test_end = n if k == n_splits - 1 else test_start + fold_sizes
        yield k, df_sorted, np.arange(0, test_start), np.arange(test_start, test_end)


def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_hat = (p >= 0.5).astype(int)
    return {
        "logloss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def simulate_roi(df: pd.DataFrame, p_home: np.ndarray, p_over: np.ndarray,
                 kelly_fraction: float = 0.25, max_stake_pct: float = 0.03,
                 min_edge: float = 0.03) -> Dict[str, float]:
    """Simulacion de negocio sobre cuotas reales historicas del feed.

    df debe traer: HomeTeamMoneyLine, AwayTeamMoneyLine, OverUnder,
    OverPayout, UnderPayout, Target_HomeWin, Target_TotalPoints.
    """
    bankroll, peak, max_dd = 100.0, 100.0, 0.0
    n_bets, wins, staked, returned = 0, 0, 0.0, 0.0
    clv_records: List[float] = []

    for i, row in enumerate(df.itertuples()):
        # --- Moneyline ---
        hm = getattr(row, "HomeTeamMoneyLine", None)
        am = getattr(row, "AwayTeamMoneyLine", None)
        if pd.notna(hm) and pd.notna(am):
            ph_mkt, pa_mkt = devig_two_way(hm, am)
            for side, p_model, odds, p_mkt in (
                ("home", p_home[i], hm, ph_mkt),
                ("away", 1 - p_home[i], am, pa_mkt),
            ):
                edge = p_model - p_mkt
                if edge < min_edge:
                    continue
                f = min(kelly_fraction * max(0.0, kelly_fraction_full(p_model, odds)),
                        max_stake_pct)
                if f <= 0:
                    continue
                stake = f * bankroll
                won = (row.Target_HomeWin == 1.0) if side == "home" else (row.Target_HomeWin == 0.0)
                pnl = stake * (american_to_decimal(odds) - 1) if won else -stake
                bankroll += pnl
                staked += stake
                returned += stake + pnl
                n_bets += 1
                wins += int(won)
                peak = max(peak, bankroll)
                max_dd = max(max_dd, (peak - bankroll) / peak)

        # --- Totales ---
        ou = getattr(row, "OverUnder", None)
        op = getattr(row, "OverPayout", None)
        up = getattr(row, "UnderPayout", None)
        if pd.notna(ou) and pd.notna(op) and pd.notna(up):
            po_mkt, pu_mkt = devig_two_way(op, up)
            for side, p_model, odds, p_mkt in (
                ("over", p_over[i], op, po_mkt),
                ("under", 1 - p_over[i], up, pu_mkt),
            ):
                edge = p_model - p_mkt
                if edge < min_edge:
                    continue
                f = min(kelly_fraction * max(0.0, kelly_fraction_full(p_model, odds)),
                        max_stake_pct)
                if f <= 0:
                    continue
                stake = f * bankroll
                total = row.Target_TotalPoints
                won = (total > ou) if side == "over" else (total < ou)
                push = total == ou
                if push:
                    continue
                pnl = stake * (american_to_decimal(odds) - 1) if won else -stake
                bankroll += pnl
                staked += stake
                returned += stake + pnl
                n_bets += 1
                wins += int(won)
                peak = max(peak, bankroll)
                max_dd = max(max_dd, (peak - bankroll) / peak)

    return {
        "n_bets": n_bets,
        "win_rate": wins / n_bets if n_bets else float("nan"),
        "roi": (returned - staked) / staked if staked else float("nan"),
        "final_bankroll": bankroll,
        "max_drawdown_pct": max_dd,
    }


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de calibracion (prob predicha vs frecuencia real por deciles)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": f"[{bins[b]:.1f},{bins[b+1]:.1f})",
                         "n": int(m.sum()),
                         "p_pred_mean": float(p[m].mean()),
                         "freq_real": float(y_true[m].mean())})
    return pd.DataFrame(rows)
