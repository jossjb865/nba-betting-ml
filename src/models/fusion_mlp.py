"""Capa de fusion (feature fusion): MLP meta-modelo con calibracion.

Estrategia de ensamble (stacking):
  Nivel 0 (modelos base):
    - CatBoost Moneyline      -> p_cb_ml  (prob. victoria local)
    - LSTM Momentum           -> p_lstm_ml, total_lstm
    - Poisson Bivariada       -> p_bp_ml, p_bp_over (a la linea), total_bp
  Nivel 1 (meta-modelo MLP):
    Entrada = [p_cb_ml, p_lstm_ml, p_bp_ml, total_lstm, total_bp,
               linea_mercado (OverUnder), diferencial de totales vs linea]
    Salida  = probabilidad final calibrada (isotonica o Platt) para Moneyline
              y total final ajustado para Over/Under.

Las predicciones de nivel 0 para entrenar el meta-modelo se generan con
validacion cruzada temporal (walk-forward) para evitar leakage: el meta-modelo
nunca ve predicciones in-sample de los modelos base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False


if _TORCH_OK:

    class _FusionNet(nn.Module):
        def __init__(self, n_in: int, hidden=(64, 32), dropout: float = 0.30):
            super().__init__()
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.body = nn.Sequential(*layers)
            self.win_head = nn.Linear(prev, 1)
            self.total_head = nn.Linear(prev, 1)

        def forward(self, x):
            h = self.body(x)
            return self.win_head(h).squeeze(-1), self.total_head(h).squeeze(-1)


@dataclass
class FusionConfig:
    hidden_sizes: Tuple[int, ...] = (64, 32)
    dropout: float = 0.30
    learning_rate: float = 1e-3
    epochs: int = 60
    patience: int = 8
    calibration: str = "isotonic"   # isotonic | sigmoid | none
    total_mean: float = 0.0
    total_std: float = 1.0


class FusionMLP:
    META_FEATURES = ["p_cb_ml", "p_lstm_ml", "p_bp_ml",
                     "total_lstm", "total_bp", "market_total_line", "diff_total_vs_line"]

    def __init__(self, config: Optional[FusionConfig] = None, device: Optional[str] = None):
        self.cfg = config or FusionConfig()
        self.device = device or ("cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu")
        self.net = None
        self.calibrator = None

    # ------------------------------------------------------------------ #
    def _to_matrix(self, df) -> np.ndarray:
        X = df[self.META_FEATURES].to_numpy(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        return X

    # ------------------------------------------------------------------ #
    def fit(self, meta_df, y_win: np.ndarray, y_total: np.ndarray) -> Dict[str, list]:
        """meta_df: DataFrame con META_FEATURES generados out-of-fold."""
        if not _TORCH_OK:
            raise RuntimeError("PyTorch no instalado")
        torch.manual_seed(42)
        X = self._to_matrix(meta_df)
        self.cfg.total_mean = float(np.nanmean(y_total))
        self.cfg.total_std = float(np.nanstd(y_total)) or 1.0
        yt_z = (y_total - self.cfg.total_mean) / self.cfg.total_std

        n = len(X)
        idx = np.arange(n)
        split = int(n * 0.85)
        tr, va = idx[:split], idx[split:]   # split temporal (datos ya ordenados por fecha)

        self.net = _FusionNet(X.shape[1], tuple(self.cfg.hidden_sizes),
                              self.cfg.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)
        bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()

        X_t = torch.from_numpy(X).to(self.device)
        yw_t = torch.from_numpy(y_win.astype(np.float32)).to(self.device)
        yt_t = torch.from_numpy(yt_z.astype(np.float32)).to(self.device)

        best_val, best_state, bad = np.inf, None, 0
        hist = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.cfg.epochs + 1):
            self.net.train()
            logit, total = self.net(X_t[tr])
            loss = bce(logit, yw_t[tr]) + 0.5 * mse(total, yt_t[tr])
            opt.zero_grad(); loss.backward(); opt.step()
            hist["train_loss"].append(float(loss.item()))

            self.net.eval()
            with torch.no_grad():
                lv, tv = self.net(X_t[va])
                vloss = bce(lv, yw_t[va]) + 0.5 * mse(tv, yt_t[va])
            hist["val_loss"].append(float(vloss.item()))
            if vloss.item() < best_val - 1e-4:
                best_val, best_state, bad = vloss.item(), \
                    {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

        # --- Calibracion sobre el tramo de validacion temporal ---
        raw = self._raw_win_proba(X[va])
        if self.cfg.calibration == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw, y_win[va])
        elif self.cfg.calibration == "sigmoid":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw.reshape(-1, 1), y_win[va])
        else:
            self.calibrator = None
        logger.info("Fusion MLP entrenada. Calibracion: %s", self.cfg.calibration)
        return hist

    # ------------------------------------------------------------------ #
    def _raw_win_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logit, _ = self.net(torch.from_numpy(X.astype(np.float32)).to(self.device))
        return torch.sigmoid(logit).cpu().numpy()

    def predict_proba(self, meta_df) -> np.ndarray:
        X = self._to_matrix(meta_df)
        raw = self._raw_win_proba(X)
        if self.calibrator is None:
            return raw
        if isinstance(self.calibrator, IsotonicRegression):
            return self.calibrator.predict(raw)
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict_total(self, meta_df) -> np.ndarray:
        X = self._to_matrix(meta_df)
        self.net.eval()
        with torch.no_grad():
            _, total = self.net(torch.from_numpy(X.astype(np.float32)).to(self.device))
        return total.cpu().numpy() * self.cfg.total_std + self.cfg.total_mean
