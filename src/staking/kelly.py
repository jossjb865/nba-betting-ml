"""Capa de gestion de riesgo y staking: Criterio de Kelly + Valor Esperado.

Formulas (cuotas en formato americano, nativas de SportsDataIO):
- Conversion:  american > 0 -> dec = 1 + a/100 ; american < 0 -> dec = 1 + 100/|a|
- Probabilidad implicita (sin vig, normalizacion proporcional / "devig"):
    p_imp = 1/dec ; p_devig = p_imp / sum(p_imp del mercado)
- Valor Esperado por unidad:  EV = p * (dec - 1) - (1 - p)
- Kelly completo:             f* = (b*p - q) / b   con b = dec - 1, q = 1 - p
- Kelly fraccionado:          f  = kelly_fraction * f*  (p.ej. 0.25 = Quarter-Kelly)
- Topes operativos: stake <= max_stake_pct del bankroll; solo se apuesta si
  edge >= min_edge y EV >= min_ev.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def american_to_decimal(american: float) -> float:
    a = float(american)
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def implied_probability(american: float) -> float:
    return 1.0 / american_to_decimal(american)


def devig_two_way(american_a: float, american_b: float) -> tuple:
    """Elimina el margen de la casa (vig) por normalizacion proporcional."""
    pa, pb = implied_probability(american_a), implied_probability(american_b)
    s = pa + pb
    return pa / s, pb / s


def expected_value(p_model: float, american_odds: float) -> float:
    """EV neto por unidad apostada."""
    b = american_to_decimal(american_odds) - 1.0
    return p_model * b - (1.0 - p_model)


def kelly_fraction_full(p_model: float, american_odds: float) -> float:
    """Fraccion de Kelly completa. f* = (b*p - q)/b."""
    b = american_to_decimal(american_odds) - 1.0
    q = 1.0 - p_model
    return (b * p_model - q) / b


@dataclass
class StakeDecision:
    market: str                    # "Moneyline" | "Total Over" | "Total Under"
    selection: str                 # p.ej. "BOS" o "Over 224.5"
    p_model: float
    p_market_devig: float
    edge: float
    ev: float
    american_odds: float
    kelly_full: float
    kelly_applied: float           # tras fraccionar
    stake_units: float             # en unidades de bankroll
    stake_pct: float
    bet: bool                      # pasa todos los filtros
    reason: str = ""


class StakingEngine:
    def __init__(self, bankroll_units: float = 100.0, kelly_fraction: float = 0.25,
                 max_stake_pct: float = 0.03, min_edge: float = 0.03,
                 min_ev: float = 0.02):
        self.bankroll = bankroll_units
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.min_edge = min_edge
        self.min_ev = min_ev

    # ------------------------------------------------------------------ #
    def evaluate(self, market: str, selection: str, p_model: float,
                 american_odds: float, p_market_devig: float) -> StakeDecision:
        p_model = float(np.clip(p_model, 1e-6, 1 - 1e-6))
        edge = p_model - p_market_devig
        ev = expected_value(p_model, american_odds)
        f_full = max(0.0, kelly_fraction_full(p_model, american_odds))
        f_applied = self.kelly_fraction * f_full
        f_applied = min(f_applied, self.max_stake_pct)
        stake_units = round(f_applied * self.bankroll, 2)

        bet, reason = True, ""
        if f_full <= 0:
            bet, reason = False, "Kelly <= 0 (sin edge matematico)"
        elif edge < self.min_edge:
            bet, reason = False, f"edge {edge:.3f} < min {self.min_edge}"
        elif ev < self.min_ev:
            bet, reason = False, f"EV {ev:.3f} < min {self.min_ev}"
        elif stake_units <= 0:
            bet, reason = False, "stake calculado = 0"

        return StakeDecision(market, selection, p_model, p_market_devig, edge, ev,
                             american_odds, f_full, f_applied, stake_units,
                             f_applied, bet, reason)

    # ------------------------------------------------------------------ #
    def evaluate_game(self, game_label: str, p_home_win: float, p_over: float,
                      home_ml: Optional[float], away_ml: Optional[float],
                      total_line: Optional[float], over_odds: Optional[float],
                      under_odds: Optional[float]) -> list:
        """Evalua los 4 mercados de un partido y devuelve las decisiones."""
        decisions = []
        if home_ml is not None and away_ml is not None:
            ph, pa = devig_two_way(home_ml, away_ml)
            decisions.append(self.evaluate("Moneyline", f"{game_label} | LOCAL gana",
                                           p_home_win, home_ml, ph))
            decisions.append(self.evaluate("Moneyline", f"{game_label} | VISITANTE gana",
                                           1 - p_home_win, away_ml, pa))
        if total_line is not None and over_odds is not None and under_odds is not None:
            po, pu = devig_two_way(over_odds, under_odds)
            decisions.append(self.evaluate("Total", f"{game_label} | Over {total_line}",
                                           p_over, over_odds, po))
            decisions.append(self.evaluate("Total", f"{game_label} | Under {total_line}",
                                           1 - p_over, under_odds, pu))
        return decisions
