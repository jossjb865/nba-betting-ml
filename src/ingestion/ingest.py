"""Capa de ingestion: descarga los feeds reales de SportsDataIO y los persiste en Parquet.

Estructura de almacenamiento (data/raw/):
  games_{season}.parquet            <- Games by Season (calendario + marcadores + lineas pregame)
  boxscores_{date}.parquet          <- TeamGame por fecha
  player_games_{date}.parquet       <- PlayerGame por fecha
  team_season_{season}.parquet      <- TeamSeasonStats
  injuries_{date}.parquet           <- InjuredPlayers (snapshot diario)
  lineups_{date}.parquet            <- StartingLineupsByDate
  odds_markets_{gameid}.parquet     <- BettingMarketsByGameID (aplanado a nivel outcome)

Todo lo que persiste esta capa proviene exclusivamente de la API; no hay fixtures
ni datos sinteticos en ningun punto del flujo.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..sportsdataio.client import SportsDataIOClient
from ..sportsdataio.endpoints import (
    Betting,
    Players,
    Scores,
    Stats,
    fmt_date,
)

logger = logging.getLogger(__name__)


class Ingestor:
    def __init__(self, client: SportsDataIOClient, raw_dir: str = "data/raw") -> None:
        self.client = client
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Calendario y marcadores
    # ------------------------------------------------------------------ #
    def fetch_season_games(self, season: int, season_types: List[int] = (1, 3)) -> pd.DataFrame:
        """Games by Season: calendario completo con GameID, equipos, estadio,
        lineas pregame de consenso y marcadores finales."""
        path = Scores.GAMES_BY_SEASON.path_template.format(season=season)
        rows = self.client.get_json_list(path)
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("Games by Season %d devolvio 0 filas", season)
            return df
        df = df[df["SeasonType"].isin(list(season_types))].copy()
        out = self.raw_dir / f"games_{season}.parquet"
        df.to_parquet(out, index=False)
        logger.info("games_%s.parquet: %d partidos", season, len(df))
        return df

    def fetch_games_by_date(self, d: date) -> pd.DataFrame:
        """Games by Date (Live & Final): incluye PointSpread, OverUnder y moneylines."""
        path = Scores.GAMES_BY_DATE.path_template.format(date=fmt_date(d))
        return pd.DataFrame(self.client.get_json_list(path))

    # ------------------------------------------------------------------ #
    # Box scores (postgame -> entrenamiento)
    # ------------------------------------------------------------------ #
    def fetch_team_games_by_date(self, d: date) -> pd.DataFrame:
        """Team Game Stats by Date: una fila por equipo-partido con el box score agregado."""
        path = Stats.TEAM_GAME_STATS_BY_DATE.path_template.format(date=fmt_date(d))
        rows = self.client.get_json_list(path)
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(self.raw_dir / f"team_games_{d.isoformat()}.parquet", index=False)
        return df

    def fetch_player_games_by_date(self, d: date) -> pd.DataFrame:
        """Player Game Stats by Date: box score por jugador (base del calculo de impacto de bajas)."""
        path = Stats.PLAYER_GAME_STATS_BY_DATE.path_template.format(date=fmt_date(d))
        rows = self.client.get_json_list(path)
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(self.raw_dir / f"player_games_{d.isoformat()}.parquet", index=False)
        return df

    def fetch_team_season_stats(self, season: int) -> pd.DataFrame:
        path = Stats.TEAM_SEASON_STATS.path_template.format(season=season)
        df = pd.DataFrame(self.client.get_json_list(path))
        if not df.empty:
            df.to_parquet(self.raw_dir / f"team_season_{season}.parquet", index=False)
        return df

    def fetch_player_season_stats(self, season: int) -> pd.DataFrame:
        path = Stats.PLAYER_SEASON_STATS.path_template.format(season=season)
        df = pd.DataFrame(self.client.get_json_list(path))
        if not df.empty:
            df.to_parquet(self.raw_dir / f"player_season_{season}.parquet", index=False)
        return df

    # ------------------------------------------------------------------ #
    # Lesiones y lineups (pregame)
    # ------------------------------------------------------------------ #
    def fetch_injuries(self, snapshot_date: Optional[date] = None) -> pd.DataFrame:
        """Player Details - by Injured: InjuryStatus (Out/Doubtful/...), InjuryBodyPart,
        InjuryStartDate, InjuryNotes. Snapshot diario para trazabilidad."""
        df = pd.DataFrame(self.client.get_json_list(Players.INJURED_PLAYERS.path_template))
        if not df.empty:
            d = snapshot_date or date.today()
            df["SnapshotDate"] = d.isoformat()
            df.to_parquet(self.raw_dir / f"injuries_{d.isoformat()}.parquet", index=False)
        return df

    def fetch_starting_lineups(self, d: date) -> pd.DataFrame:
        """Starting Lineups by Date: LineupStatus (Active/Inactive) y LineupConfirmed."""
        path = Players.STARTING_LINEUPS_BY_DATE.path_template.format(date=fmt_date(d))
        df = pd.DataFrame(self.client.get_json_list(path))
        if not df.empty:
            df.to_parquet(self.raw_dir / f"lineups_{d.isoformat()}.parquet", index=False)
        return df

    # ------------------------------------------------------------------ #
    # Odds (pregame)
    # ------------------------------------------------------------------ #
    def fetch_betting_markets(self, game_id: int) -> pd.DataFrame:
        """Betting Markets by GameID aplanado: una fila por (mercado, sportsbook, outcome).

        Columnas resultantes: BettingMarketID, BettingBetTypeID, BettingBetType,
        BettingPeriodTypeID, SportsbookName, BettingOutcomeType, Value,
        PayoutAmerican, PayoutDecimal, IsAvailable, Updated.
        """
        path = Betting.BETTING_MARKETS_BY_GAME_ID.path_template.format(gameid=game_id)
        markets = self.client.get_json_list(path, params={"include": "available"})
        rows: List[Dict] = []
        for m in markets or []:
            base = {
                "BettingMarketID": m.get("BettingMarketID"),
                "BettingBetTypeID": m.get("BettingBetTypeID"),
                "BettingBetType": m.get("BettingBetType"),
                "BettingPeriodTypeID": m.get("BettingPeriodTypeID"),
                "MarketName": m.get("Name"),
                "MarketUpdated": m.get("Updated"),
            }
            sportsbooks = m.get("SportsBooks") or m.get("Sportsbooks") or []
            # Estructura tolerante: algunos mercados traen Sportsbook + BettingOutcomes directos
            outcomes_blocks = []
            for sb in sportsbooks:
                outcomes_blocks.append((sb.get("Name"), sb.get("BettingOutcomes") or []))
            if not outcomes_blocks and m.get("BettingOutcomes"):
                sb_name = (m.get("Sportsbook") or {}).get("Name")
                outcomes_blocks.append((sb_name, m.get("BettingOutcomes")))
            for sb_name, outcomes in outcomes_blocks:
                for o in outcomes:
                    rows.append({
                        **base,
                        "SportsbookName": sb_name,
                        "BettingOutcomeType": o.get("BettingOutcomeType"),
                        "Value": o.get("Value"),
                        "PayoutAmerican": o.get("PayoutAmerican"),
                        "PayoutDecimal": o.get("PayoutDecimal"),
                        "IsAvailable": o.get("IsAvailable"),
                        "OutcomeUpdated": o.get("Updated"),
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["GameID"] = game_id
            df.to_parquet(self.raw_dir / f"odds_markets_{game_id}.parquet", index=False)
        return df

    # ------------------------------------------------------------------ #
    # Backfill historico (postgame -> dataset de entrenamiento)
    # ------------------------------------------------------------------ #
    def backfill_season(self, season: int, season_types: List[int] = (1, 3)) -> pd.DataFrame:
        """Descarga el calendario de una temporada y, para cada fecha con partidos
        Final, baja los TeamGame y PlayerGame correspondientes."""
        games = self.fetch_season_games(season, season_types)
        if games.empty:
            return games
        finals = games[games["Status"].isin(["Final", "F/OT"])]
        fechas = sorted({pd.to_datetime(x).date() for x in finals["Day"].dropna()})
        logger.info("Backfill %d: %d fechas con partidos finalizados", season, len(fechas))
        for d in fechas:
            tg_path = self.raw_dir / f"team_games_{d.isoformat()}.parquet"
            pg_path = self.raw_dir / f"player_games_{d.isoformat()}.parquet"
            if not tg_path.exists():
                self.fetch_team_games_by_date(d)
            if not pg_path.exists():
                self.fetch_player_games_by_date(d)
        return games

    def load_all_team_games(self) -> pd.DataFrame:
        """Concatena todos los snapshots team_games_*.parquet en un unico DataFrame."""
        frames = [pd.read_parquet(p) for p in sorted(self.raw_dir.glob("team_games_*.parquet"))]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=["StatID"]).reset_index(drop=True)

    def load_all_player_games(self) -> pd.DataFrame:
        frames = [pd.read_parquet(p) for p in sorted(self.raw_dir.glob("player_games_*.parquet"))]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=["StatID"]).reset_index(drop=True)

    def load_all_games(self) -> pd.DataFrame:
        frames = [pd.read_parquet(p) for p in sorted(self.raw_dir.glob("games_*.parquet"))]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=["GameID"]).reset_index(drop=True)


def daterange(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)
