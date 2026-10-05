"""Esquemas (mapeo de campos reales) de las entidades de SportsDataIO usadas por el sistema.

Cada dataclass documenta los campos exactos que devuelve la API y que el pipeline
consume. No se inventan campos: si un campo no existe en el feed, no se usa.

Fuentes de verdad:
- Game[]:        scores/json/GamesByDate/{date}  y  scores/json/Games/{season}
- BoxScore:      stats/json/BoxScore/{gameid}    (anida TeamGame[] y PlayerGame[])
- TeamSeason[]:  stats/json/TeamSeasonStats/{season}
- Player[]:      projections/json/InjuredPlayers, scores/json/Players/{team}
- StartingLineups[]: projections/json/StartingLineupsByDate/{date}
- GameOdds/BettingMarket/BettingOutcome: odds/json/...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Game:
    """Entidad central de calendario/marcador (Scores feed)."""
    GameID: int
    Season: int
    SeasonType: int                       # 1 Regular, 2 Preseason, 3 Postseason, 4 All-Star
    Status: str                           # Scheduled | InProgress | Final | F/OT | Postponed | Canceled
    Day: str                              # fecha (medianoche local del venue)
    DateTime: Optional[str]               # hora local del venue
    DateTimeUTC: Optional[str]
    AwayTeam: str                         # abreviatura, p.ej. "BOS"
    HomeTeam: str
    AwayTeamID: int
    HomeTeamID: int
    StadiumID: Optional[int]
    GlobalGameID: Optional[int]
    # Marcadores (postgame)
    AwayTeamScore: Optional[float] = None
    HomeTeamScore: Optional[float] = None
    # Lineas pregame de consenso incluidas en el propio Game
    PointSpread: Optional[float] = None           # desde la perspectiva del equipo local
    OverUnder: Optional[float] = None
    AwayTeamMoneyLine: Optional[int] = None       # formato americano
    HomeTeamMoneyLine: Optional[int] = None
    OverPayout: Optional[int] = None
    UnderPayout: Optional[int] = None
    PointSpreadAwayTeamMoneyLine: Optional[int] = None
    PointSpreadHomeTeamMoneyLine: Optional[int] = None
    IsClosed: Optional[bool] = None
    NeutralVenue: Optional[bool] = None
    GameEndDateTime: Optional[str] = None


@dataclass
class TeamGame:
    """Linea de box score a nivel equipo (seccion TeamGames de BoxScore)."""
    StatID: int
    TeamID: int
    Team: str
    GameID: int
    HomeOrAway: str                       # "HOME" | "AWAY"
    Wins: int
    Losses: int
    Points: float
    FieldGoalsMade: float
    FieldGoalsAttempted: float
    FieldGoalsPercentage: Optional[float]
    ThreePointersMade: float
    ThreePointersAttempted: float
    FreeThrowsMade: float
    FreeThrowsAttempted: float
    OffensiveRebounds: float
    DefensiveRebounds: float
    Rebounds: float
    Assists: float
    Steals: float
    BlockedShots: float
    Turnovers: float
    PersonalFouls: float
    PointsInThePaint: Optional[float] = None
    FastBreakPoints: Optional[float] = None
    SecondChancePoints: Optional[float] = None
    TurnoverPoints: Optional[float] = None
    Possessions: Optional[float] = None   # disponible en box scores detallados
    TrueShootingPercentage: Optional[float] = None
    EffectiveFieldGoalsPercentage: Optional[float] = None
    OffensiveRating: Optional[float] = None
    DefensiveRating: Optional[float] = None
    PlusMinus: Optional[float] = None


@dataclass
class PlayerGame:
    """Linea de box score a nivel jugador (seccion PlayerGames de BoxScore)."""
    StatID: int
    PlayerID: int
    Name: str
    Team: str
    TeamID: int
    GameID: int
    Position: Optional[str]
    Started: Optional[int]                # 1 si fue titular
    Minutes: Optional[float]
    Points: float
    FieldGoalsMade: float
    FieldGoalsAttempted: float
    ThreePointersMade: float
    ThreePointersAttempted: float
    FreeThrowsMade: float
    FreeThrowsAttempted: float
    Rebounds: float
    Assists: float
    Steals: float
    BlockedShots: float
    Turnovers: float
    PersonalFouls: float
    PlusMinus: Optional[float]
    UsageRatePercentage: Optional[float] = None


@dataclass
class TeamSeason:
    """Totales/acumulados de temporada por equipo (TeamSeasonStats)."""
    StatID: int
    TeamID: int
    Team: str
    Season: int
    SeasonType: int
    Games: int
    Wins: int
    Losses: int
    Points: float
    Possessions: Optional[float]
    OffensiveRating: Optional[float]
    DefensiveRating: Optional[float]
    FieldGoalsPercentage: Optional[float]
    EffectiveFieldGoalsPercentage: Optional[float]
    TrueShootingPercentage: Optional[float]
    ThreePointersPercentage: Optional[float]
    Assists: float
    Rebounds: float
    Turnovers: float
    Steals: float
    BlockedShots: float


@dataclass
class InjuredPlayer:
    """Jugador lesionado (Player Details - by Injured)."""
    PlayerID: int
    FirstName: str
    LastName: str
    Team: str
    TeamID: int
    Position: Optional[str]
    Status: str                           # roster status (Active, etc.)
    InjuryStatus: Optional[str]           # Out | Doubtful | Questionable | Probable | Day-To-Day
    InjuryBodyPart: Optional[str]
    InjuryStartDate: Optional[str]
    InjuryNotes: Optional[str]


@dataclass
class StartingLineupEntry:
    """Entrada de StartingLineupsByDate (proyectado y confirmado)."""
    GameID: int
    PlayerID: int
    Name: str
    Team: str
    TeamID: int
    HomeOrAway: str
    Position: Optional[str]
    LineupStatus: Optional[str]           # Active | Inactive
    LineupConfirmed: Optional[bool]
    InjuryStatus: Optional[str] = None


@dataclass
class BettingOutcome:
    BettingOutcomeID: int
    BettingMarketID: int
    SportsBook: Optional[dict]
    BettingOutcomeTypeID: Optional[int]
    BettingOutcomeType: Optional[str]     # "Home" | "Away" | "Over" | "Under"
    Value: Optional[float]                # linea (p.ej. 224.5 en totales)
    Participant: Optional[str]
    PayoutAmerican: Optional[int]
    PayoutDecimal: Optional[float]
    IsAvailable: Optional[bool]
    Updated: Optional[str] = None


@dataclass
class BettingMarket:
    BettingMarketID: int
    BettingEventID: Optional[int]
    BettingMarketTypeID: Optional[int]
    BettingMarketType: Optional[str]      # p.ej. "Game Lines"
    BettingBetTypeID: Optional[int]       # 1 Moneyline, 2 Spread, 3 Total
    BettingBetType: Optional[str]
    BettingPeriodTypeID: Optional[int]    # 1 FullGame
    BettingPeriodType: Optional[str]
    Name: Optional[str]
    TeamID: Optional[int]
    TeamKey: Optional[str]
    Sportsbook: Optional[dict]            # {SportsbookID, Name}
    BettingOutcomes: List[dict] = field(default_factory=list)
    ConsensusOutcomes: List[dict] = field(default_factory=list)
    Created: Optional[str] = None
    Updated: Optional[str] = None
