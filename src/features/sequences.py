"""Constructor de secuencias temporales por equipo para el modulo LSTM (lstm_momentum).

Para cada partido genera dos tensores:
  X_home: (seq_len, n_feats) con los ultimos `seq_len` partidos del equipo LOCAL
  X_away: (seq_len, n_feats) idem para el VISITANTE
ordenados estrictamente por fecha y construidos solo con partidos anteriores.

Las features de cada paso temporal provienen del box score real (TeamGame):
puntos, ratings ofensivo/defensivo, eFG%, posesiones, margen de victoria,
descanso, indicador de localia. El "momentum" lo aprende la recurrencia.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEQ_FEATURES = [
    "Points", "OffensiveRating", "DefensiveRating", "Possessions",
    "EffectiveFieldGoalsPercentage", "TrueShootingPercentage",
    "Assists", "Turnovers", "Rebounds", "Margin", "IsHome", "RestDays",
]


def _prep_team_sequences(tg: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Devuelve {TeamID: DataFrame ordenado por fecha con SEQ_FEATURES + GameID}."""
    tg = tg.copy()
    tg["Margin"] = np.where(tg["IsHome"] == 1,
                            tg["Points"] - tg["OppPoints"],
                            tg["Points"] - tg["OppPoints"])
    tg["RestDays"] = tg.groupby("TeamID")["Day"].diff().dt.days - 1
    cols = ["GameID", "Day"] + [c for c in SEQ_FEATURES if c in tg.columns]
    out: Dict[int, pd.DataFrame] = {}
    for tid, d in tg.groupby("TeamID"):
        out[tid] = d.sort_values("Day")[cols].reset_index(drop=True)
    return out


class SequenceBuilder:
    def __init__(self, seq_len: int = 12):
        self.seq_len = seq_len
        self.feature_cols: List[str] = []
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def fit_normalization(self, tg: pd.DataFrame) -> None:
        self.feature_cols = [c for c in SEQ_FEATURES if c in tg.columns]
        self.mean_ = {c: float(np.nanmean(tg[c])) for c in self.feature_cols}
        self.std_ = {c: float(np.nanstd(tg[c])) or 1.0 for c in self.feature_cols}

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.feature_cols:
            df[c] = (df[c] - self.mean_[c]) / self.std_[c]
        return df.fillna(0.0)

    # ------------------------------------------------------------------ #
    def build(
        self,
        team_games_enriched: pd.DataFrame,
        games: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (X_home, X_away, game_ids) con shape (n, seq_len, n_feats).

        team_games_enriched debe incluir: TeamID, GameID, Day, IsHome, Points,
        OppPoints y las SEQ_FEATURES disponibles.
        """
        if not self.feature_cols:
            self.fit_normalization(team_games_enriched)
        seqs = _prep_team_sequences(self._normalize(team_games_enriched))
        # indice rapido: (TeamID) -> {GameID: posicion}
        pos_index = {tid: {g: i for i, g in enumerate(d["GameID"])}
                     for tid, d in seqs.items()}

        Xh, Xa, ids = [], [], []
        for _, g in games.iterrows():
            gid, hid, aid = g["GameID"], g["HomeTeamID"], g["AwayTeamID"]
            if hid not in seqs or aid not in seqs:
                continue
            ih, ia = pos_index[hid].get(gid), pos_index[aid].get(gid)
            if ih is None or ia is None:
                continue
            sh = seqs[hid].iloc[max(0, ih - self.seq_len):ih][self.feature_cols].to_numpy(np.float32)
            sa = seqs[aid].iloc[max(0, ia - self.seq_len):ia][self.feature_cols].to_numpy(np.float32)
            if len(sh) < 3 or len(sa) < 3:
                continue
            # zero-padding a la izquierda si hay menos de seq_len partidos
            if len(sh) < self.seq_len:
                sh = np.vstack([np.zeros((self.seq_len - len(sh), len(self.feature_cols)), np.float32), sh])
            if len(sa) < self.seq_len:
                sa = np.vstack([np.zeros((self.seq_len - len(sa), len(self.feature_cols)), np.float32), sa])
            Xh.append(sh); Xa.append(sa); ids.append(gid)
        logger.info("Secuencias LSTM: %d partidos, shape %s", len(ids),
                    (self.seq_len, len(self.feature_cols)))
        return np.asarray(Xh, np.float32), np.asarray(Xa, np.float32), np.asarray(ids)
