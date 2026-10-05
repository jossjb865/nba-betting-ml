"""Registro de endpoints reales de la API de SportsDataIO para NBA.

Referencia: https://sportsdata.io/developers/api-documentation/nba
Convencion de fechas del proveedor: `YYYY-MMM-DD` en ingles (p.ej. 2026-JAN-15).
Los "Call Interval" indicados son los publicados en la documentacion oficial.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


def fmt_date(d: date) -> str:
    """Formato requerido por SportsDataIO: 2026-JAN-15 (mes en ingles, mayusculas)."""
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    return f"{d.year}-{months[d.month - 1]}-{d.day:02d}"


@dataclass(frozen=True)
class Endpoint:
    path_template: str
    call_interval_hint: str
    lifecycle: str  # Pregame | Live & Final | Final


class Scores:
    """Event Feeds > Scores & Game State."""

    # Calendario completo de la temporada (incluye GameID, DateTimeUTC, estadio, canales)
    GAMES_BY_SEASON = Endpoint("scores/json/Games/{season}", "1 Hour", "Pregame/Live/Final")
    # Partidos de una fecha: incluye lineas pregame (PointSpread, OverUnder,
    # AwayTeamMoneyLine, HomeTeamMoneyLine, OverPayout, UnderPayout) y marcadores por cuarto
    GAMES_BY_DATE = Endpoint("scores/json/GamesByDate/{date}", "5 Seconds", "Live & Final")
    # Version ligera sin info de lineas
    BASIC_GAMES_BY_DATE = Endpoint("scores/json/BasicGamesByDate/{date}", "5 Seconds", "Live & Final")
    # Estadios (StadiumID -> nombre, ciudad, capacidad)
    STADIUMS = Endpoint("scores/json/Stadiums", "1 Day", "Pregame")
    # Clasificacion por temporada
    STANDINGS = Endpoint("scores/json/Standings/{season}", "1 Hour", "Live & Final")
    # ¿Hay partidos en curso ahora mismo?
    ARE_ANY_GAMES_IN_PROGRESS = Endpoint("scores/json/AreAnyGamesInProgress", "5 Seconds", "Live")


class Stats:
    """Event Feeds > Team & Player Stats."""

    # Box score completo de un partido (equipo + jugadores). Sujeto a correcciones post-game.
    BOX_SCORE = Endpoint("stats/json/BoxScore/{gameid}", "1 Minute", "Live & Final")
    BOX_SCORES_BY_DATE = Endpoint("stats/json/BoxScoresByDate/{date}", "1 Minute", "Live & Final")
    # Deltas en vivo (solo stats modificadas en los ultimos X minutos)
    BOX_SCORES_DELTA_BY_DATE = Endpoint("stats/json/BoxScoresDeltaByDate/{date}/{minutes}", "3 Seconds", "Live")
    # Stats de jugadores por fecha (PlayerGame[])
    PLAYER_GAME_STATS_BY_DATE = Endpoint("stats/json/PlayerGameStatsByDate/{date}", "5 Minutes", "Live & Final")
    # Stats de equipo por fecha (TeamGame[])
    TEAM_GAME_STATS_BY_DATE = Endpoint("stats/json/TeamGameStatsByDate/{date}", "5 Minutes", "Live & Final")
    # Totales de temporada por equipo (TeamSeason[])
    TEAM_SEASON_STATS = Endpoint("stats/json/TeamSeasonStats/{season}", "5 Minutes", "Final")
    # Totales de temporada por jugador (PlayerSeason[])
    PLAYER_SEASON_STATS = Endpoint("stats/json/PlayerSeasonStats/{season}", "15 Minutes", "Final")
    # Stats permitidos por posicion rival (TeamSeason[])
    TEAM_STATS_ALLOWED_BY_POSITION = Endpoint("stats/json/TeamSeasonStatsAllowedByPosition/{season}", "5 Minutes", "Final")
    # Game logs de un jugador (PlayerGame[])
    PLAYER_GAME_LOGS_BY_SEASON = Endpoint("stats/json/PlayerGameStatsByPlayer/{season}/{playerid}/{numberofgames}", "1 Hour", "Final")


class Players:
    """Player Feeds > Depth Charts, Lineups & Injuries."""

    # Jugadores lesionados activos (InjuryStatus, InjuryBodyPart, InjuryStartDate, InjuryNote)
    INJURED_PLAYERS = Endpoint("projections/json/InjuredPlayers", "1 Minute", "Pregame")
    # Lineups proyectados/confirmados por fecha (LineupStatus: Active/Inactive, Confirmed)
    STARTING_LINEUPS_BY_DATE = Endpoint("projections/json/StartingLineupsByDate/{date}", "3 Minutes", "Pregame")
    # Roster de un equipo con bio + estado de lesion
    PLAYERS_BY_TEAM = Endpoint("scores/json/Players/{team}", "10 Minutes", "Pregame")
    # Depth charts de toda la liga
    DEPTH_CHARTS = Endpoint("scores/json/DepthCharts", "5 Minutes", "Pregame")
    # Transacciones por fecha
    TRANSACTIONS_BY_DATE = Endpoint("scores/json/TransactionsByDate/{date}", "1 Minute", "Pregame")


class Betting:
    """Betting Feeds > Odds (feed agregado multi-sportsbook).

    Estructura: BettingEvent -> BettingMarket[] -> BettingOutcome[].
    Campos clave de BettingMarket: BettingMarketID, BettingMarketTypeID,
    BettingBetTypeID (Moneyline/Spread/Total), BettingPeriodTypeID (FullGame...),
    Sportsbook (Name, SportsbookID), BettingOutcome[] (BettingOutcomeType:
    Home/Away/Over/Under, Value, PayoutAmerican, PayoutDecimal, IsAvailable).
    """

    # Mercados de un partido concreto (include=available devuelve solo mercados activos)
    BETTING_MARKETS_BY_GAME_ID = Endpoint(
        "odds/json/BettingMarketsByGameID/{gameid}", "1 Minute", "Pregame & Live")
    # Meta-mercados de un partido (listado de mercados disponibles)
    BETTING_MARKET_METADATA_BY_GAME_ID = Endpoint(
        "odds/json/BettingMarketMetadataByGameID/{gameid}", "1 Minute", "Pregame")
    # Odds de partido por fecha (GameOdds[]: consenso por mercado)
    GAME_ODDS_BY_DATE = Endpoint("odds/json/GameOddsByDate/{date}", "1 Minute", "Pregame & Live")
    # Movimiento de linea de un mercado concreto
    BETTING_MARKET_LINE_MOVEMENT = Endpoint(
        "odds/json/BettingMarketLineMovement/{marketid}", "1 Minute", "Pregame & Live")
    # Splits de dinero/apuestas por mercado
    BETTING_SPLITS_BY_MARKET = Endpoint(
        "odds/json/BettingMarketSplitsByMarket/{marketid}", "15 Minutes", "Pregame")
    # Enumeraciones de tipos (BettingBetTypeID, BettingPeriodTypeID, ...)
    ODDS_TYPES = Endpoint("odds/json/BettingMetadata", "1 Day", "Reference")


# Identificadores de tipos de apuesta documentados para mercados de partido completo
BET_TYPE_MONEYLINE = 1
BET_TYPE_SPREAD = 2
BET_TYPE_TOTAL = 3
PERIOD_FULL_GAME = 1
