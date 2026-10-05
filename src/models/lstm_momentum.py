"""Modulo LSTM con momentum (lstm_momentum).

Arquitectura: dos torres LSTM (una por equipo) que procesan las secuencias de
los ultimos N partidos reales del equipo (box scores + descanso + localia).
Las salidas se concatenan y alimentan dos cabezas:
  - Clasificacion: P(gana local)   -> mercado Moneyline
  - Regresion:     total esperado  -> mercado Over/Under

El "momentum" no es una feature manual: lo captura la recurrencia al modelar
la trayectoria temporal de rendimiento (rachas, desgaste por calendario via
RestDays/GamesLast7d como pasos de la secuencia, tendencias de ritmo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False
    logger.warning("PyTorch no disponible: lstm_momentum quedara deshabilitado.")


if _TORCH_OK:

    class _TeamTower(nn.Module):
        def __init__(self, n_feats: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_feats, hidden_size=hidden,
                                num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)

        def forward(self, x):  # x: (batch, seq, feats)
            out, (h_n, _) = self.lstm(x)
            return h_n[-1]     # (batch, hidden): estado final = resumen del momentum


    class LSTMMomentumNet(nn.Module):
        def __init__(self, n_feats: int, hidden: int = 64, layers: int = 2,
                     dropout: float = 0.30):
            super().__init__()
            self.home_tower = _TeamTower(n_feats, hidden, layers, dropout)
            self.away_tower = _TeamTower(n_feats, hidden, layers, dropout)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            )
            self.win_head = nn.Linear(32, 1)     # logit victoria local
            self.total_head = nn.Linear(32, 1)   # total esperado (desestandarizado fuera)

        def forward(self, xh, xa):
            z = torch.cat([self.home_tower(xh), self.away_tower(xa)], dim=1)
            h = self.head(z)
            return self.win_head(h).squeeze(-1), self.total_head(h).squeeze(-1)


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.30
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 40
    patience: int = 6
    total_mean: float = 0.0
    total_std: float = 1.0


class LSTMMomentum:
    def __init__(self, config: Optional[LSTMConfig] = None, device: Optional[str] = None):
        self.cfg = config or LSTMConfig()
        self.device = device or ("cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu")
        self.net = None

    # ------------------------------------------------------------------ #
    def fit(self, Xh: np.ndarray, Xa: np.ndarray,
            y_win: np.ndarray, y_total: np.ndarray,
            val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
            ) -> Dict[str, list]:
        if not _TORCH_OK:
            raise RuntimeError("PyTorch no instalado")
        torch.manual_seed(42)
        n_feats = Xh.shape[2]
        self.cfg.total_mean = float(y_total.mean())
        self.cfg.total_std = float(y_total.std()) or 1.0
        y_total_z = (y_total - self.cfg.total_mean) / self.cfg.total_std

        ds = TensorDataset(torch.from_numpy(Xh), torch.from_numpy(Xa),
                           torch.from_numpy(y_win.astype(np.float32)),
                           torch.from_numpy(y_total_z.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        self.net = LSTMMomentumNet(n_feats, self.cfg.hidden_size,
                                   self.cfg.num_layers, self.cfg.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val, best_state, bad = np.inf, None, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.net.train()
            tot = 0.0
            for xh, xa, yw, yt in dl:
                xh, xa, yw, yt = (t.to(self.device) for t in (xh, xa, yw, yt))
                logit, total = self.net(xh, xa)
                loss = bce(logit, yw) + 0.5 * mse(total, yt)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(yw)
            history["train_loss"].append(tot / len(ds))

            if val is not None:
                vl = self._val_loss(val, bce, mse)
                history["val_loss"].append(vl)
                if vl < best_val - 1e-4:
                    best_val, best_state, bad = vl, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= self.cfg.patience:
                        logger.info("Early stopping en epoch %d (val_loss=%.4f)", epoch, best_val)
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return history

    def _val_loss(self, val, bce, mse) -> float:
        Xh, Xa, yw, yt = val
        yt_z = (yt - self.cfg.total_mean) / self.cfg.total_std
        self.net.eval()
        with torch.no_grad():
            logit, total = self.net(torch.from_numpy(Xh).to(self.device),
                                    torch.from_numpy(Xa).to(self.device))
            loss = bce(logit, torch.from_numpy(yw.astype(np.float32)).to(self.device)) \
                 + 0.5 * mse(total, torch.from_numpy(yt_z.astype(np.float32)).to(self.device))
        return float(loss.item())

    # ------------------------------------------------------------------ #
    def predict_proba(self, Xh: np.ndarray, Xa: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logit, _ = self.net(torch.from_numpy(Xh).to(self.device),
                                torch.from_numpy(Xa).to(self.device))
            return torch.sigmoid(logit).cpu().numpy()

    def predict_total(self, Xh: np.ndarray, Xa: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            _, total = self.net(torch.from_numpy(Xh).to(self.device),
                                torch.from_numpy(Xa).to(self.device))
            return total.cpu().numpy() * self.cfg.total_std + self.cfg.total_mean

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict(), "cfg": self.cfg.__dict__,
                    "n_feats": self.net.home_tower.lstm.input_size}, path)

    def load(self, path: str, n_feats: int) -> None:
        obj = torch.load(path, map_location=self.device, weights_only=False)
        self.cfg = LSTMConfig(**obj["cfg"])
        self.net = LSTMMomentumNet(n_feats, self.cfg.hidden_size,
                                   self.cfg.num_layers, self.cfg.dropout).to(self.device)
        self.net.load_state_dict(obj["state_dict"])
