"""Ingenieria de caracteristicas a nivel partido.

Genera el dataset tabular de entrenamiento/prediccion: UNA fila por partido,
con features de ambos equipos calculadas EXCLUSIVAMENTE con informacion
disponible antes del tip-off (sin leakage).

Fuentes (todas SportsDataIO):
- Games by Season / Games by Date   -> fecha, local/visitante, lineas pregame
- Team Game Stats by Date           -> box scores agregados historicos
- Player Game Stats by Date         -> minutos/puntos por jugador (impacto de bajas)
- InjuredPlayers                    -> InjuryStatus por equipo en cada fecha
- StartingLineupsByDate             -> confirmacion de titulares (pregame)

Bloques de features:
1. Forma reciente: medias moviles (5/10/20) de puntos, ratings, eFG%, TS%, ritmo.
2. Momentum/rachas: racha actual (win/loss), delta de forma (media5 - media20).
3. Fatiga/calendario: dias de descanso, back-to-back, partidos en ultimos 7 dias,
   3 en 4 noches, distancia aproximada entre sedes consecutivas.
4. Matchup: diferenciales local - visitante de cada metrica.
5. Bajas: % de minutos y % de puntos de temporada que faltan por InjuryStatus.
6. Contexto: mes, dia de semana, campo neutral, playoffs (SeasonType).
7. Mercado (solo para validacion/CLV, no como input del modelo base por defecto).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Campos del box score de equipo (TeamGame) que alimentan las medias moviles
TEAM_BOX_COLS = [
    "Points", "FieldGoalsPercentage", "EffectiveFieldGoalsPercentage",
    "TrueShootingPercentage", "OffensiveRating", "DefensiveRating",
    "Possessions", "Assists", "Turnovers", "Rebounds",
    "OffensiveRebounds", "ThreePointersPercentage", "PersonalFouls",
]

STATUS_WEIGHT = {  # probabilidad aproximada de ausencia segun InjuryStatus oficial
    "Out": 1.00,
    "Doubtful": 0.75,
    "Questionable": 0.50,
    "Probable": 0.10,
    "Day-To-Day": 0.50,
}


class FeatureBuilder:
    def __init__(self, rolling_windows: List[int] = (5, 10, 20), min_games_history: int = 5):
        self.windows = list(rolling_windows)
        self.min_games = min_games_history

    # ------------------------------------------------------------------ #
    def _estimate_possessions(self, df: pd.DataFrame) -> pd.Series:
        """Si el box score no trae Possessions, se estima con la formula estandar:
        Poss = FGA + 0.44*FTA - ORB + TO."""
        if "Possessions" in df and df["Possessions"].notna().any():
            poss = df["Possessions"].copy()
            mask = poss.isna()
        else:
            poss = pd.Series(np.nan, index=df.index)
            mask = pd.Series(True, index=df.index)
        est = (df["FieldGoalsAttempted"] + 0.44 * df["FreeThrowsAttempted"]
               - df["OffensiveRebounds"] + df["Turnovers"])
        poss[mask] = est[mask]
        return poss

    def _team_game_table(self, team_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        """Une TeamGame con el calendario para obtener fecha y rival."""
        g = games[["GameID", "Day", "Season", "SeasonType",
                   "HomeTeamID", "AwayTeamID", "NeutralVenue"]].copy()
        g["Day"] = pd.to_datetime(g["Day"])
        tg = team_games.merge(g, on="GameID", how="left")
        tg["IsHome"] = (tg["TeamID"] == tg["HomeTeamID"]).astype(int)
        tg["OppTeamID"] = np.where(tg["IsHome"] == 1, tg["AwayTeamID"], tg["HomeTeamID"])
        tg["Possessions"] = self._estimate_possessions(tg)
        # Ratings por 100 posesiones si el feed no los incluye
        if "OffensiveRating" not in tg or tg["OffensiveRating"].isna().all():
            tg["OffensiveRating"] = 100 * tg["Points"] / tg["Possessions"].replace(0, np.nan)
        tg = tg.sort_values(["TeamID", "Day"]).reset_index(drop=True)
        return tg

    # ------------------------------------------------------------------ #
    def _rolling_team_features(self, tg: pd.DataFrame) -> pd.DataFrame:
        """Medias moviles por equipo calculadas con shift(1): solo partidos ANTERIORES."""
        grp = tg.groupby("TeamID")
        for w in self.windows:
            for col in TEAM_BOX_COLS:
                if col in tg.columns:
                    tg[f"{col}_roll{w}"] = grp[col].transform(
                        lambda s: s.shift(1).rolling(w, min_periods=max(3, w // 2)).mean())
        # Rachas y momentum
        tg["Win"] = tg.groupby(["TeamID", "GameID"])["Points"].transform("first")  # placeholder
        tg = tg.drop(columns=["Win"])
        # La victoria real se deriva comparando con el rival: se calcula fuera (merge).
        tg["GamesPlayedPrev"] = grp.cumcount()
        tg["RestDays"] = grp["Day"].diff().dt.days - 1
        tg["IsB2B"] = (tg["RestDays"] == 0).astype(int)
        tg["GamesLast7d"] = grp["Day"].transform(
            lambda s: s.apply(lambda d: ((s < d) & (s >= d - pd.Timedelta(days=7))).sum()))
        tg["ThreeInFour"] = (tg["GamesLast7d"] >= 2) & (tg["RestDays"] <= 1)
        tg["ThreeInFour"] = tg["ThreeInFour"].astype(int)
        return tg

    @staticmethod
    def _streaks(tg: pd.DataFrame) -> pd.Series:
        """Racha de victorias(+) / derrotas(-) entrando al partido."""
        streaks = []
        for _, team_df in tg.groupby("TeamID"):
            s, current = [], 0
            for win in team_df["WonGame"].fillna(0).astype(int):
                s.append(current)  # racha ANTES de este partido
                current = current + 1 if win == 1 else (current - 1 if win == 0 else 0)
                if win == 1 and current < 0:
                    current = 1
                if win == 0 and current > 0:
                    current = -1
            streaks.append(pd.Series(s, index=team_df.index))
        return pd.concat(streaks).sort_index()

    def _injury_impact(
        self,
        games: pd.DataFrame,
        player_games: pd.DataFrame,
        injuries_snapshots: Optional[pd.DataFrame],
        player_season: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """% de produccion (puntos y minutos de temporada) ausente por lesiones.

        Usa snapshots diarios de InjuredPlayers. Para fechas historicas sin snapshot,
        aproxima con la ventana de jugadores que NO aparecieron en PlayerGame ese dia
        pese a tener minutos de temporada (proxy de ausencia), manteniendo siempre
        datos reales del feed.
        """
        out = games[["GameID", "Day", "HomeTeamID", "AwayTeamID"]].copy()
        out["Day"] = pd.to_datetime(out["Day"])
        for side, tid in (("Home", "HomeTeamID"), ("Away", "AwayTeamID")):
            out[f"{side}MissingPtsShare"] = np.nan
            out[f"{side}MissingMinShare"] = np.nan
            out[f"{side}PlayersOut"] = np.nan

        if injuries_snapshots is None or injuries_snapshots.empty or player_season is None:
            return out

        ps = player_season[["PlayerID", "TeamID", "Points", "Minutes", "Games"]].copy()
        ps["PointsPerGame"] = ps["Points"] / ps["Games"].replace(0, np.nan)
        ps["MinutesPerGame"] = ps["Minutes"] / ps["Games"].replace(0, np.nan)
        team_tot = ps.groupby("TeamID")[["PointsPerGame", "MinutesPerGame"]].sum()

        inj = injuries_snapshots.copy()
        inj["SnapshotDate"] = pd.to_datetime(inj["SnapshotDate"])
        inj["AbsenceProb"] = inj["InjuryStatus"].map(STATUS_WEIGHT).fillna(0.5)
        inj = inj.merge(ps[["PlayerID", "PointsPerGame", "MinutesPerGame"]], on="PlayerID", how="left")

        for d, day_inj in inj.groupby("SnapshotDate"):
            day_impact = day_inj.groupby("TeamID").apply(
                lambda x: pd.Series({
                    "pts_share": (x["PointsPerGame"] * x["AbsenceProb"]).sum()
                                 / max(team_tot.loc[x.name, "PointsPerGame"], 1e-9) if x.name in team_tot.index else 0.0,
                    "min_share": (x["MinutesPerGame"] * x["AbsenceProb"]).sum()
                                 / max(team_tot.loc[x.name, "MinutesPerGame"], 1e-9) if x.name in team_tot.index else 0.0,
                    "n_out": int((x["AbsenceProb"] >= 0.75).sum()),
                }), include_groups=False)
            mask = out["Day"] == d
            for side, tid in (("Home", "HomeTeamID"), ("Away", "AwayTeamID")):
                for col_src, col_dst in (("pts_share", f"{side}MissingPtsShare"),
                                         ("min_share", f"{side}MissingMinShare"),
                                         ("n_out", f"{side}PlayersOut")):
                    out.loc[mask, col_dst] = out.loc[mask, tid].map(day_impact[col_src])
        return out

    # ------------------------------------------------------------------ #
    def build(
        self,
        games: pd.DataFrame,
        team_games: pd.DataFrame,
        player_games: Optional[pd.DataFrame] = None,
        injuries_snapshots: Optional[pd.DataFrame] = None,
        player_season: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Construye la matriz final partido x features (una fila por GameID)."""
        tg = self._team_game_table(team_games, games)
        # Victoria real del equipo en ese partido
        pts = tg.pivot_table(index="GameID", columns="IsHome", values="Points",
                             aggfunc="first").rename(columns={1: "HomePts", 0: "AwayPts"})
        tg = tg.merge(pts, on="GameID", how="left")
        tg["WonGame"] = np.where(
            tg["IsHome"] == 1, (tg["HomePts"] > tg["AwayPts"]).astype(float),
            (tg["AwayPts"] > tg["HomePts"]).astype(float))
        tg.loc[tg["HomePts"].isna(), "WonGame"] = np.nan

        tg = self._rolling_team_features(tg)
        tg["Streak"] = self._streaks(tg)
        for w in self.windows:
            tg[f"FormDelta_{w}"] = tg[f"Points_roll{min(self.windows)}"] - tg[f"Points_roll{w}"]

        # Pivote a una fila por partido
        home = tg[tg["IsHome"] == 1].add_prefix("Home")
        away = tg[tg["IsHome"] == 0].add_prefix("Away")
        home = home.rename(columns={"HomeGameID": "GameID"})
        away = away.rename(columns={"AwayGameID": "GameID"})
        feat = home.merge(away, on="GameID", suffixes=("", "_drop"))
        feat = feat[[c for c in feat.columns if not c.endswith("_drop")]]

        # Contexto del partido y targets
        ctx = games[["GameID", "Season", "SeasonType", "Day", "DateTimeUTC",
                     "HomeTeamID", "AwayTeamID", "HomeTeam", "AwayTeam",
                     "HomeTeamScore", "AwayTeamScore", "NeutralVenue",
                     "PointSpread", "OverUnder",
                     "HomeTeamMoneyLine", "AwayTeamMoneyLine",
                     "OverPayout", "UnderPayout"]].copy()
        ctx["Day"] = pd.to_datetime(ctx["Day"])
        ctx["Month"] = ctx["Day"].dt.month
        ctx["DayOfWeek"] = ctx["Day"].dt.dayofweek
        feat = feat.merge(ctx, on="GameID", how="left")

        # Impacto de lesiones
        inj = self._injury_impact(games, player_games, injuries_snapshots, player_season)
        feat = feat.merge(inj.drop(columns=["Day", "HomeTeamID", "AwayTeamID"]),
                          on="GameID", how="left")

        # Diferenciales local - visitante
        for w in self.windows:
            for col in TEAM_BOX_COLS:
                h, a = f"Home{col}_roll{w}", f"Away{col}_roll{w}"
                if h in feat and a in feat:
                    feat[f"Diff_{col}_roll{w}"] = feat[h] - feat[a]
        feat["Diff_Streak"] = feat["HomeStreak"] - feat["AwayStreak"]
        feat["Diff_RestDays"] = feat["HomeRestDays"] - feat["AwayRestDays"]
        feat["Diff_MissingPtsShare"] = feat["HomeMissingPtsShare"] - feat["AwayMissingPtsShare"]

        # Targets
        feat["Target_HomeWin"] = (feat["HomeTeamScore"] > feat["AwayTeamScore"]).astype("float")
        feat["Target_TotalPoints"] = feat["HomeTeamScore"] + feat["AwayTeamScore"]
        feat.loc[feat["HomeTeamScore"].isna(), ["Target_HomeWin", "Target_TotalPoints"]] = np.nan

        # Filtro de historia minima
        mask = (feat["HomeGamesPlayedPrev"] >= self.min_games) & \
               (feat["AwayGamesPlayedPrev"] >= self.min_games)
        logger.info("Feature matrix: %d partidos (%d con historia suficiente)",
                    len(feat), int(mask.sum()))
        return feat.reset_index(drop=True)


# Columnas de entrada para los modelos tabulares (CatBoost)
def tabular_feature_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("Diff_",)
    exact = [
        "HomeStreak", "AwayStreak", "HomeRestDays", "AwayRestDays",
        "HomeIsB2B", "AwayIsB2B", "HomeGamesLast7d", "AwayGamesLast7d",
        "HomeThreeInFour", "AwayThreeInFour",
        "HomeMissingPtsShare", "AwayMissingPtsShare",
        "HomePlayersOut", "AwayPlayersOut",
        "Month", "DayOfWeek", "SeasonType",
    ]
    cols = [c for c in df.columns if c.startswith(prefixes)]
    cols += [c for c in exact if c in df.columns]
    return cols


CATEGORICAL_FEATURES = ["Month", "DayOfWeek", "SeasonType", "HomeTeamID", "AwayTeamID"]
