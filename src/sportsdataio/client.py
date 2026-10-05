"""Cliente HTTP para la API oficial de SportsDataIO (NBA).

Base URL: https://api.sportsdata.io/v3/nba
Autenticacion: query param `key` (Subscription Key de SportsDataIO).

Reglas implementadas:
- Reintentos con backoff exponencial ante errores transitorios (429/5xx).
- Rate limiting minimo entre llamadas para respetar los "Call Interval"
  documentados por endpoint (p.ej. Games by Date: 5s, Box Scores: 1 min).
- Sin datos simulados: este cliente SOLO habla con la API real.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SportsDataIOError(RuntimeError):
    """Error generico de la capa de acceso a SportsDataIO."""


class SportsDataIOClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.sportsdata.io/v3/nba",
        timeout: int = 30,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        min_call_interval: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("SPORTSDATAIO_API_KEY", "")
        if not self.api_key:
            raise SportsDataIOError(
                "Falta SPORTSDATAIO_API_KEY. Exportala como variable de entorno "
                "o defini el secreto en GitHub Actions."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.min_call_interval = min_call_interval
        self._last_call_ts = 0.0
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET con reintentos. `path` es relativo a base_url (p.ej. 'scores/json/GamesByDate/2026-JAN-15')."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        query: Dict[str, Any] = dict(params or {})
        query["key"] = self.api_key

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=query, timeout=self.timeout)
                self._last_call_ts = time.monotonic()
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self.backoff_seconds * (2 ** (attempt - 1))
                    logger.warning("HTTP %s en %s (intento %d/%d). Reintento en %.1fs",
                                   resp.status_code, path, attempt, self.max_retries, wait)
                    time.sleep(wait)
                    continue
                raise SportsDataIOError(
                    f"HTTP {resp.status_code} en {path}: {resp.text[:300]}"
                )
            except requests.RequestException as exc:  # red / timeout
                last_exc = exc
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning("Error de red en %s (intento %d/%d): %s",
                               path, attempt, self.max_retries, exc)
                time.sleep(wait)
        raise SportsDataIOError(f"Fallo definitivo llamando a {path}: {last_exc}")

    # ------------------------------------------------------------------ #
    def get_json_list(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        data = self.get(path, params=params)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        raise SportsDataIOError(f"Respuesta inesperada (no lista) en {path}: {type(data)}")
