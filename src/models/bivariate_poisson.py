"""Modelo de Distribucion de Poisson Bivariada para marcadores NBA.

Modelo (Karlis & Ntzoufras, 2003):
  X = Y1 + Y3   (puntos del equipo local)
  Y = Y2 + Y3   (puntos del equipo visitante)
  Y1 ~ Poisson(lambda1), Y2 ~ Poisson(lambda2), Y3 ~ Poisson(lambda3)
  cov(X, Y) = lambda3  -> captura la correlacion positiva entre marcadores
  (prorrogas, ritmo de partido compartido, garbage time).

lambda_home = exp(intercept + home_adv + attack_home + defence_away)
lambda_away = exp(intercept           + attack_away + defence_home)

Estimacion: maxima verosimilitud sobre los marcadores historicos reales
(HomeTeamScore / AwayTeamScore de Games by Season), con shrinkage hacia la
media de liga para estabilizar equipos con pocos partidos.

Aplicaciones directas a mercados:
- Moneyline: P(home gana) = sum_{x>y} P(X=x, Y=y) + 0.5*P(empate reglamentario->OT handled)
- Totales:   P(X+Y > linea) integrando la distribucion del total.
Nota NBA: los empates se resuelven en prorroga, por lo que la probabilidad de
empate se reparte proporcionalmente entre ambos equipos (empate -> 50/50 en OT).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)

MAX_GOALS_GRID = 220  # soporte del grid de marcadores (los puntos NBA rara vez superan 180)


@dataclass
class PoissonFitResult:
    team_params: Dict[int, Dict[str, float]]
    intercept: float
    home_advantage: float
    lambda3: float
    n_games: int
    loglik: float


class BivariatePoissonModel:
    def __init__(self, lambda3_bounds: Tuple[float, float] = (0.0, 8.0), shrinkage: float = 0.10):
        self.lambda3_bounds = lambda3_bounds
        self.shrinkage = shrinkage
        self.fit_: Optional[PoissonFitResult] = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def pmf(x: np.ndarray, y: np.ndarray, l1: float, l2: float, l3: float) -> np.ndarray:
        """PMF bivariada P(X=x, Y=y) con sumas finitas estables (log-space)."""
        x = np.atleast_1d(x).astype(int)
        y = np.atleast_1d(y).astype(int)
        out = np.zeros((len(x), len(y)))
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                k_max = min(xi, yj)
                if k_max < 0:
                    continue
                ks = np.arange(0, k_max + 1)
                # sum_k Pois(xi-k; l1) * Pois(yj-k; l2) * Pois(k; l3), en log-espacio
                log_terms = (
                    (-l1 + np.log(l1) * (xi - ks) - gammaln(xi - ks + 1))
                    + (-l2 + np.log(l2) * (yj - ks) - gammaln(yj - ks + 1))
                    + (-l3 + np.log(max(l3, 1e-12)) * ks - gammaln(ks + 1))
                )
                m = log_terms.max()
                out[i, j] = np.exp(m) * np.exp(log_terms - m).sum()
        return out

    # ------------------------------------------------------------------ #
    def fit(self, games: pd.DataFrame) -> PoissonFitResult:
        """games: columnas HomeTeamID, AwayTeamID, HomeTeamScore, AwayTeamScore (Final)."""
        df = games.dropna(subset=["HomeTeamScore", "AwayTeamScore"]).copy()
        if len(df) < 200:
            raise ValueError(f"Insuficientes partidos para ajustar Poisson bivariada: {len(df)}")
        teams = sorted(set(df["HomeTeamID"]) | set(df["AwayTeamID"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hs = df["HomeTeamScore"].to_numpy(float)
        as_ = df["AwayTeamScore"].to_numpy(float)
        home_idx = df["HomeTeamID"].map(tidx).to_numpy(int)
        away_idx = df["AwayTeamID"].map(tidx).to_numpy(int)
        lg_home = np.log(np.clip(hs, 1e-6, None))
        lg_away = np.log(np.clip(as_, 1e-6, None))
        league_mean_log = 0.5 * (lg_home.mean() + lg_away.mean())

        def unpack(theta):
            intercept, home_adv = theta[0], theta[1]
            attack = theta[2:2 + n]
            defence = theta[2 + n:2 + 2 * n]
            l3 = self.lambda3_bounds[0] + (self.lambda3_bounds[1] - self.lambda3_bounds[0]) / (1 + np.exp(-theta[2 + 2 * n]))
            return intercept, home_adv, attack, defence, l3

        def neg_ll(theta):
            intercept, home_adv, attack, defence, l3 = unpack(theta)
            l1 = np.exp(intercept + home_adv + attack[home_idx] + defence[away_idx])
            l2 = np.exp(intercept + attack[away_idx] + defence[home_idx])
            # log-verosimilitud bivariada, evaluada por lotes con la pmf de arriba
            ll = 0.0
            for i in range(len(df)):
                p = self.pmf(np.array([int(hs[i])]), np.array([int(as_[i])]), l1[i], l2[i], l3)
                ll += np.log(max(p[0, 0], 1e-300))
            # shrinkage (prior gaussiano hacia 0 para attack/defence)
            ll -= self.shrinkage * 50.0 * (np.sum(attack ** 2) + np.sum(defence ** 2))
            return -ll

        theta0 = np.zeros(2 + 2 * n + 1)
        theta0[0] = league_mean_log
        theta0[1] = 0.05
        theta0[-1] = -2.0  # lambda3 pequeno de arranque
        logger.info("Ajustando Poisson bivariada sobre %d partidos, %d equipos...", len(df), n)
        res = minimize(neg_ll, theta0, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-8})
        intercept, home_adv, attack, defence, l3 = unpack(res.x)
        team_params = {
            t: {"attack": float(attack[tidx[t]]), "defence": float(defence[tidx[t]])}
            for t in teams
        }
        self.fit_ = PoissonFitResult(team_params, float(intercept), float(home_adv),
                                     float(l3), int(len(df)), float(-res.fun))
        logger.info("Poisson bivariada ajustada. lambda3=%.3f home_adv=%.4f loglik=%.1f",
                    l3, home_adv, -res.fun)
        return self.fit_

    # ------------------------------------------------------------------ #
    def predict_lambdas(self, home_team_id: int, away_team_id: int) -> Tuple[float, float, float]:
        assert self.fit_ is not None, "Modelo no ajustado"
        f = self.fit_
        hp = f.team_params.get(home_team_id, {"attack": 0.0, "defence": 0.0})
        ap = f.team_params.get(away_team_id, {"attack": 0.0, "defence": 0.0})
        l_home = float(np.exp(f.intercept + f.home_advantage + hp["attack"] + ap["defence"]))
        l_away = float(np.exp(f.intercept + ap["attack"] + hp["defence"]))
        return l_home, l_away, f.lambda3

    def score_matrix(self, home_team_id: int, away_team_id: int,
                     max_points: int = MAX_GOALS_GRID) -> np.ndarray:
        """Matriz (max_points+1)^2 con P(X=i, Y=j)."""
        l1, l2, l3 = self.predict_lambdas(home_team_id, away_team_id)
        grid = np.arange(0, max_points + 1)
        return self.pmf(grid, grid, l1, l2, l3)

    def predict_markets(self, home_team_id: int, away_team_id: int,
                        total_line: Optional[float] = None) -> Dict[str, float]:
        """Probabilidades de mercado derivadas de la matriz de marcadores."""
        M = self.score_matrix(home_team_id, away_team_id)
        # M[i,j] = P(local=i, visitante=j): victoria local => i > j => triu estricta
        p_home = np.triu(M, 1).sum()
        p_away = np.tril(M, -1).sum()
        p_draw = np.trace(M)
        # Reparto de empate 50/50 (prorroga simetrica como prior)
        p_home += 0.5 * p_draw
        p_away += 0.5 * p_draw

        out = {"P_home_win": float(p_home), "P_away_win": float(p_away),
               "P_draw_regulation": float(p_draw)}
        if total_line is not None:
            total_mass = np.zeros(2 * M.shape[0] + 1)
            for i in range(M.shape[0]):
                total_mass[i:i + M.shape[1]] += M[i, :]
            cutoff = int(np.floor(total_line)) + 1
            out["P_over"] = float(total_mass[cutoff:].sum())
            out["P_under"] = float(total_mass[:cutoff].sum())
            out["ExpectedTotal"] = float((np.arange(len(total_mass)) * total_mass).sum())
        return out
