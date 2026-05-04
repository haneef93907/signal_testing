#!/usr/bin/env python3
"""
Binance Spot USDT Signal Scanner

Scans all Spot USDT pairs and returns high-probability signals using a custom 0-10 score.

Requirements covered:
- Multi-timeframe klines: 5m (entry), 15m (confirmation), 1h (trend)
- Indicators: EMA50/EMA200, RSI14, Volume MA20, 5-candle momentum
- Risk/Reward and profit potential calculation
- Score filter >= 8 and additional quality filters
- Optional email alerts
- Optional scheduler (every 5 minutes)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import smtplib
import sys
import time
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.utils import formatdate
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

try:
    from binance.spot import Spot
except ImportError:
    Spot = None  # type: ignore


# ---------------------------- Config ----------------------------
TIMEFRAMES = {
    "entry": "5m",
    "confirm": "15m",
    "trend": "1h",
}
CANDLE_LIMIT = 250  # >=200 as requested; extra buffer for indicator warmup

MIN_SCORE = 8
MIN_PROFIT_FILTER = 5.0
MAX_PROFIT_FILTER = 25.0

RETRY_SLEEP_SECONDS = 2
MAX_RETRIES = 3
DEFAULT_MAX_WORKERS = 10
DEFAULT_REQUEST_SLEEP_SECONDS = 0.0

# Email alert defaults (can still be overridden by env vars)
DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_USER = "haneef93907@gmail.com"
DEFAULT_SMTP_PASSWORD = "gutgoqkysvzlxmsx"
DEFAULT_ALERT_EMAIL_FROM = "haneef93907@gmail.com"
DEFAULT_ALERT_EMAIL_TO = "haneef93907@gmail.com"


# ---------------------------- Errors ----------------------------
class RestrictedLocationError(Exception):
    """Raised when Binance blocks requests from the current server region."""


# ---------------------------- Data Models ----------------------------
@dataclass
class Signal:
    symbol: str
    price: float
    buy_price: float
    expected_sell: float
    resistance_touches: int
    stop_loss: float
    score: int
    trend: str
    trend_24h: str
    trend_7d: str
    btc_regime: str
    volume_strength: str
    momentum_strength: str
    rsi: float
    williams_r: float
    wr_status: str
    support: float
    volume_24h_usdt: float
    probability: int
    hold_time_text: str
    risk_level: str
    final_verdict: str
    rr: float
    profit_pct: float


# ---------------------------- Scanner ----------------------------
class BinanceSignalScanner:
    def __init__(self, client: Spot, max_workers: int = DEFAULT_MAX_WORKERS, request_sleep_seconds: float = DEFAULT_REQUEST_SLEEP_SECONDS):
        self.client = client
        self.max_workers = max(1, max_workers)
        self.request_sleep_seconds = max(0.0, request_sleep_seconds)
        self.btc_regime = "Neutral"

    @staticmethod
    def _is_restricted_location_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "restricted location" in msg or "(451" in msg

    def _safe_request(self, fn, *args, **kwargs):
        """Simple retry wrapper for transient API issues / rate-limit spikes."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if self._is_restricted_location_error(exc):
                    raise RestrictedLocationError(str(exc)) from exc
                if attempt == MAX_RETRIES:
                    raise
                logging.warning("Request failed (%s/%s): %s", attempt, MAX_RETRIES, exc)
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    def get_usdt_symbols(self) -> List[str]:
        info = self._safe_request(self.client.exchange_info)
        symbols = []
        for s in info.get("symbols", []):
            if (
                s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
                and s.get("isSpotTradingAllowed", False)
            ):
                symbols.append(s["symbol"])
        return symbols

    def fetch_klines_df(self, symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
        rows = self._safe_request(
            self.client.klines,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "num_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.astype("float64").fillna(50.0)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        highest_high = df["high"].rolling(period).max()
        lowest_low = df["low"].rolling(period).min()
        wr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan)
        return wr.fillna(-50.0)

    @staticmethod
    def classify_volume(current_volume: float, avg_volume: float) -> str:
        if avg_volume <= 0:
            return "weak"
        if current_volume > 1.5 * avg_volume:
            return "strong"
        if current_volume >= avg_volume:
            return "moderate"
        return "weak"

    @staticmethod
    def classify_momentum(price_change_pct: float) -> str:
        if price_change_pct > 2:
            return "strong"
        if 1 <= price_change_pct <= 2:
            return "moderate"
        return "weak"

    @staticmethod
    def classify_trend(change_pct: float, bull_threshold: float, bear_threshold: float) -> str:
        if change_pct >= bull_threshold:
            return "Bullish"
        if change_pct <= bear_threshold:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def classify_wr_status(wr_value: float) -> str:
        if wr_value <= -80:
            return "Oversold"
        if wr_value >= -20:
            return "Overbought"
        return "Neutral"

    @staticmethod
    def classify_risk_level(rr: float, volume_strength: str, momentum_strength: str) -> str:
        if rr >= 2 and volume_strength in {"strong", "moderate"} and momentum_strength in {"strong", "moderate"}:
            return "Low"
        if rr >= 1.3 and volume_strength != "weak":
            return "Medium"
        return "High"

    @staticmethod
    def estimate_probability(score: int, rr: float, rsi_value: float) -> int:
        prob = 45 + (score * 4)
        if rr >= 2:
            prob += 4
        if 45 <= rsi_value <= 60:
            prob += 2
        return int(max(40, min(92, prob)))

    @staticmethod
    def estimate_hold_time(current_price: float, target_price: float, atr_15m: float) -> str:
        if atr_15m <= 0:
            return "~N/A (ATR-estimated)"
        distance = max(0.0, target_price - current_price)
        candles = distance / atr_15m
        hours = (candles * 15.0) / 60.0
        low_h = max(1, int(round(hours * 0.75)))
        high_h = max(low_h, int(round(hours * 1.25)))
        return f"~{low_h}-{high_h} h (ATR-estimated)"

    def get_btc_regime(self) -> str:
        try:
            btc_df = self.fetch_klines_df("BTCUSDT", TIMEFRAMES["trend"])
            if len(btc_df) < 210:
                return "Neutral"
            ema50 = self.ema(btc_df["close"], 50).iloc[-1]
            ema200 = self.ema(btc_df["close"], 200).iloc[-1]
            return "Bull" if ema50 > ema200 else "Bear"
        except Exception:
            return "Neutral"

    @staticmethod
    def score_signal(
        bullish_trend: bool,
        volume_strength: str,
        momentum_strength: str,
        rr: float,
        profit_pct: float,
        rsi_value: float,
    ) -> int:
        score = 0

        if bullish_trend:
            score += 2

        if volume_strength == "strong":
            score += 2
        elif volume_strength == "moderate":
            score += 1

        if momentum_strength == "strong":
            score += 2
        elif momentum_strength == "moderate":
            score += 1

        if rr >= 3:
            score += 2
        elif 2 <= rr < 3:
            score += 1

        if 5 <= profit_pct <= 15:
            score += 1

        if 50 <= rsi_value <= 65:
            score += 1

        return score

    def analyze_symbol(self, symbol: str) -> Optional[Signal]:
        try:
            df_5m = self.fetch_klines_df(symbol, TIMEFRAMES["entry"])
            df_15m = self.fetch_klines_df(symbol, TIMEFRAMES["confirm"])
            df_1h = self.fetch_klines_df(symbol, TIMEFRAMES["trend"])

            # Ensure enough data for indicators/support-resistance window
            if min(len(df_5m), len(df_15m), len(df_1h)) < 210:
                return None

            close_1h = df_1h["close"]
            ema50_1h = self.ema(close_1h, 50).iloc[-1]
            ema200_1h = self.ema(close_1h, 200).iloc[-1]
            bullish_trend = ema50_1h > ema200_1h

            close_15m = df_15m["close"]
            rsi_15m = float(self.rsi(close_15m, 14).iloc[-1])
            wr_15m = float(self.williams_r(df_15m, 14).iloc[-1])
            wr_status = self.classify_wr_status(wr_15m)
            atr_15m = float(self.atr(df_15m, 14).iloc[-1])

            current_volume_5m = float(df_5m["volume"].iloc[-1])
            avg_volume_20_5m = float(df_5m["volume"].tail(20).mean())
            volume_strength = self.classify_volume(current_volume_5m, avg_volume_20_5m)

            close_5m = df_5m["close"]
            current_price = float(close_5m.iloc[-1])
            reference_price = float(close_5m.iloc[-6])
            price_change_pct = ((current_price - reference_price) / reference_price) * 100
            momentum_strength = self.classify_momentum(price_change_pct)

            support = float(df_15m["low"].tail(20).min())
            resistance = float(df_15m["high"].tail(20).max())
            resistance_touches = int((df_15m["high"].tail(20) >= resistance * 0.995).sum())

            if current_price <= support:
                return None

            rr = (resistance - current_price) / (current_price - support)
            profit_pct = ((resistance - current_price) / current_price) * 100
            stop_loss = support * 0.99
            hold_time_text = self.estimate_hold_time(current_price, resistance, atr_15m)

            # 24h / 7d trends from 1h closes
            close_1h_series = df_1h["close"]
            close_24h_ago = float(close_1h_series.iloc[-25])
            close_7d_ago = float(close_1h_series.iloc[-169])
            trend_24h_pct = ((current_price - close_24h_ago) / close_24h_ago) * 100
            trend_7d_pct = ((current_price - close_7d_ago) / close_7d_ago) * 100
            trend_24h = self.classify_trend(trend_24h_pct, bull_threshold=2.0, bear_threshold=-2.0)
            trend_7d = self.classify_trend(trend_7d_pct, bull_threshold=5.0, bear_threshold=-5.0)

            score = self.score_signal(
                bullish_trend=bullish_trend,
                volume_strength=volume_strength,
                momentum_strength=momentum_strength,
                rr=rr,
                profit_pct=profit_pct,
                rsi_value=rsi_15m,
            )

            # Hard filters
            if score < MIN_SCORE:
                return None
            if volume_strength == "weak" or momentum_strength == "weak":
                return None
            if profit_pct <= MIN_PROFIT_FILTER:
                return None
            if profit_pct > MAX_PROFIT_FILTER:
                return None
            if rr <= 0:
                return None

            risk_level = self.classify_risk_level(rr, volume_strength, momentum_strength)
            probability = self.estimate_probability(score, rr, rsi_15m)
            final_verdict = "BUY"

            volume_24h_usdt = 0.0
            try:
                ticker_24h = self._safe_request(self.client.ticker_24hr, symbol=symbol)
                volume_24h_usdt = float(ticker_24h.get("quoteVolume", 0.0))
            except Exception:
                volume_24h_usdt = 0.0

            return Signal(
                symbol=symbol,
                price=current_price,
                buy_price=current_price,
                expected_sell=resistance,
                resistance_touches=resistance_touches,
                stop_loss=stop_loss,
                score=score,
                trend="bullish" if bullish_trend else "bearish",
                trend_24h=trend_24h,
                trend_7d=trend_7d,
                btc_regime=self.btc_regime,
                volume_strength=volume_strength,
                momentum_strength=momentum_strength,
                rsi=rsi_15m,
                williams_r=wr_15m,
                wr_status=wr_status,
                support=support,
                volume_24h_usdt=volume_24h_usdt,
                probability=probability,
                hold_time_text=hold_time_text,
                risk_level=risk_level,
                final_verdict=final_verdict,
                rr=rr,
                profit_pct=profit_pct,
            )

        except Exception as exc:
            logging.debug("Skipping %s due to error: %s", symbol, exc)
            return None

    def scan(self) -> List[Signal]:
        symbols = self.get_usdt_symbols()
        self.btc_regime = self.get_btc_regime()
        logging.info("Scanning %d USDT symbols with %d workers...", len(symbols), self.max_workers)

        signals: List[Signal] = []
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analyze_symbol, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                completed += 1
                signal = future.result()
                if signal:
                    signals.append(signal)

                if completed % 50 == 0 or completed == len(symbols):
                    logging.info("Progress: %d/%d", completed, len(symbols))

                if self.request_sleep_seconds > 0:
                    time.sleep(self.request_sleep_seconds)

        signals.sort(key=lambda s: (s.score, s.rr, s.profit_pct), reverse=True)
        return signals


# ---------------------------- Output ----------------------------
def print_signals(signals: List[Signal]) -> None:
    if not signals:
        print(f"No high-quality signals found (score >= {MIN_SCORE}).")
        return

    for s in signals:
        touches_label = "2+ touches" if s.resistance_touches >= 2 else "1 touch"
        volume_24h_m = s.volume_24h_usdt / 1_000_000
        print(
            f"Coin:               {s.symbol}\n"
            f"Current Price:      {s.price:.6f}\n"
            f"Buy Price:          {s.buy_price:.6f}\n"
            f"Expected Sell:      {s.expected_sell:.6f} ({touches_label})\n"
            f"Stop-Loss:          {s.stop_loss:.6f}\n"
            f"Expected Profit:    {s.profit_pct:.2f}%\n"
            f"Risk/Reward Ratio:  {s.rr:.2f}:1\n"
            f"Probability:        {s.probability}%\n"
            f"Est. Hold Time:     {s.hold_time_text}\n"
            f"WR Status:          {s.wr_status} (W%R={s.williams_r:.0f})\n"
            f"RSI-14:             {s.rsi:.1f}\n"
            f"24h Trend:          {s.trend_24h}\n"
            f"7d Trend:           {s.trend_7d}\n"
            f"BTC Regime:         {s.btc_regime}\n"
            f"Support:            {s.support:.6f}\n"
            f"24h Volume:         {volume_24h_m:.2f}M USDT\n"
            f"Volume:             {s.volume_strength.capitalize()}\n"
            f"Momentum:           {s.momentum_strength.capitalize()}\n"
            f"Risk Level:         {s.risk_level}\n"
            f"Final Verdict:      {s.final_verdict}\n"
            f"{'-' * 55}"
        )


def build_alert_message(signals: List[Signal], max_items: int = 20) -> str:
    lines = [f"Binance Signal Scanner Alerts (Score >= {MIN_SCORE})", ""]
    for s in signals[:max_items]:
        touches_label = "2+ touches" if s.resistance_touches >= 2 else "1 touch"
        volume_24h_m = s.volume_24h_usdt / 1_000_000
        lines.append(
            f"Coin:               {s.symbol}\n"
            f"Current Price:      {s.price:.6f}\n"
            f"Buy Price:          {s.buy_price:.6f}\n"
            f"Expected Sell:      {s.expected_sell:.6f} ({touches_label})\n"
            f"Stop-Loss:          {s.stop_loss:.6f}\n"
            f"Expected Profit:    {s.profit_pct:.2f}%\n"
            f"Risk/Reward Ratio:  {s.rr:.2f}:1\n"
            f"Probability:        {s.probability}%\n"
            f"Est. Hold Time:     {s.hold_time_text}\n"
            f"WR Status:          {s.wr_status} (W%R={s.williams_r:.0f})\n"
            f"RSI-14:             {s.rsi:.1f}\n"
            f"24h Trend:          {s.trend_24h}\n"
            f"7d Trend:           {s.trend_7d}\n"
            f"BTC Regime:         {s.btc_regime}\n"
            f"Support:            {s.support:.6f}\n"
            f"24h Volume:         {volume_24h_m:.2f}M USDT\n"
            f"Volume:             {s.volume_strength.capitalize()}\n"
            f"Momentum:           {s.momentum_strength.capitalize()}\n"
            f"Risk Level:         {s.risk_level}\n"
            f"Final Verdict:      {s.final_verdict}\n"
            f"{'-' * 55}"
        )
    return "\n".join(lines)


def send_email_alerts(
    signals: List[Signal],
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    email_from: str,
    email_to: str,
) -> None:
    if not signals:
        return

    body = build_alert_message(signals)
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"Binance Signals: {len(signals)} match(es)"
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Date"] = formatdate(localtime=True)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, [email_to], msg.as_string())
        logging.info("Email alert sent.")
    except Exception as exc:
        logging.error("Failed to send email alert: %s", exc)


# ---------------------------- Runtime ----------------------------
def run_once(scanner: BinanceSignalScanner, email: bool) -> None:
    start = time.time()
    try:
        signals = scanner.scan()
    except RestrictedLocationError as exc:
        logging.error(
            "Binance blocked this server location (HTTP 451). "
            "Deploy in a Binance-allowed region or use another exchange endpoint. Details: %s",
            exc,
        )
        return

    print_signals(signals)

    if email:
        smtp_host = os.getenv("SMTP_HOST", DEFAULT_SMTP_HOST).strip()
        smtp_port = int(os.getenv("SMTP_PORT", str(DEFAULT_SMTP_PORT)).strip() or str(DEFAULT_SMTP_PORT))
        smtp_user = os.getenv("SMTP_USER", DEFAULT_SMTP_USER).strip()
        smtp_password = os.getenv("SMTP_PASSWORD", DEFAULT_SMTP_PASSWORD).strip()
        email_from = os.getenv("ALERT_EMAIL_FROM", DEFAULT_ALERT_EMAIL_FROM).strip()
        email_to = os.getenv("ALERT_EMAIL_TO", DEFAULT_ALERT_EMAIL_TO).strip()
        if smtp_host and smtp_user and smtp_password and email_from and email_to:
            send_email_alerts(
                signals=signals,
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                smtp_user=smtp_user,
                smtp_password=smtp_password,
                email_from=email_from,
                email_to=email_to,
            )
        else:
            logging.warning(
                "Email enabled but SMTP settings are incomplete."
            )

    elapsed = time.time() - start
    logging.info("Scan completed in %.1f seconds. Found %d signals.", elapsed, len(signals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance Spot USDT Signal Scanner")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=0,
        help="If > 0, run continuously every N seconds (e.g., 300 for 5m scheduler).",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help=(
            "Send alerts to email via SMTP env vars: SMTP_HOST, SMTP_USER, SMTP_PASSWORD, "
            "ALERT_EMAIL_TO (optional: SMTP_PORT, ALERT_EMAIL_FROM)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel workers for symbol scanning (higher = faster, but more API pressure).",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=DEFAULT_REQUEST_SLEEP_SECONDS,
        help="Optional small delay after each completed symbol in parallel mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if Spot is None:
        logging.error("Missing dependency: binance-connector. Install with: pip install binance-connector")
        return 1

    client = Spot()
    scanner = BinanceSignalScanner(
        client,
        max_workers=args.workers,
        request_sleep_seconds=args.request_sleep,
    )

    if args.interval_seconds > 0:
        logging.info("Scheduler mode: running every %d seconds", args.interval_seconds)
        while True:
            try:
                run_once(scanner, email=args.email)
            except KeyboardInterrupt:
                logging.info("Stopped by user.")
                return 0
            except Exception as exc:
                logging.exception("Unexpected top-level error: %s", exc)
            time.sleep(args.interval_seconds)
    else:
        run_once(scanner, email=args.email)

    return 0


if __name__ == "__main__":
    sys.exit(main())
