"""Notificaciones Telegram via Bot API oficial.

Autenticacion: bot token de @BotFather + chat_id destino.
Sin dependencias externas: se usa requests contra
https://api.telegram.org/bot<TOKEN>/sendMessage
"""

from __future__ import annotations

import logging
import os
from typing import List

import requests

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 parse_mode: str = "Markdown", max_len: int = 4000):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.parse_mode = parse_mode
        self.max_len = max_len
        if not self.token or not self.chat_id:
            raise RuntimeError("Faltan TELEGRAM_BOT_TOKEN y/o TELEGRAM_CHAT_ID.")

    # ------------------------------------------------------------------ #
    def send(self, text: str) -> bool:
        """Envia un mensaje, troceando si excede el limite de Telegram."""
        ok = True
        for chunk in self._split(text):
            resp = requests.post(
                API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": chunk,
                      "parse_mode": self.parse_mode,
                      "disable_web_page_preview": True},
                timeout=20,
            )
            if resp.status_code != 200:
                logger.error("Telegram %s: %s", resp.status_code, resp.text[:300])
                ok = False
        return ok

    def _split(self, text: str) -> List[str]:
        if len(text) <= self.max_len:
            return [text]
        chunks, current = [], ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > self.max_len:
                chunks.append(current)
                current = ""
            current += line + "\n"
        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------ #
    @staticmethod
    def format_predictions(date_str: str, picks: List[dict],
                           all_games: List[dict]) -> str:
        """Genera el mensaje diario.

        picks: decisiones con bet=True (value bets), cada una un dict con
               market, selection, p_model, edge, ev, american_odds, stake_units.
        all_games: resumen de todos los partidos evaluados.
        """
        lines = [f"*NBA Model - {date_str}*", ""]
        if picks:
            lines.append(f"*Value bets detectadas: {len(picks)}*")
            for p in picks:
                lines.append(
                    f"\n*{'-'*28}*\n"
                    f"*Mercado:* {p['market']}\n"
                    f"*Seleccion:* {p['selection']}\n"
                    f"*Prob. modelo:* {p['p_model']:.1%} | *Cuota:* {p['american_odds']:+.0f}\n"
                    f"*Edge:* {p['edge']:.1%} | *EV:* {p['ev']:.1%}\n"
                    f"*Stake (Kelly/4):* {p['stake_units']:.2f} u"
                )
        else:
            lines.append("_Sin value bets hoy (filtros de edge/EV no superados)._")
        if all_games:
            lines.append("\n*Resumen de partidos evaluados:*")
            for g in all_games:
                lines.append(
                    f"- {g['away']} @ {g['home']}: "
                    f"P(local)={g['p_home']:.0%}, total={g['total']:.1f} "
                    f"(linea {g.get('line', 'n/d')})"
                )
        lines.append("\n_Aviso: contenido informativo, no es asesoria financiera._")
        return "\n".join(lines)
