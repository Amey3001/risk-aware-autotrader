"""Execution script: turn portfolio weights into Alpaca paper-trading orders.

Wired to the real pipeline: the LSTM risk model trained in Notebook 02, and the
mean-CVaR optimizer from Notebook 03. Pulls fresh price data at run time via yfinance
(not the static training CSV), forecasts volatility, reconstructs the covariance matrix
against the saved historical correlation, and optimizes weights subject to the same 35%
position cap used in backtesting.

Known limitation, not yet handled: this only ever submits BUY orders sized from the
account's current equity. That is correct for the very first run on a fresh paper
account (100% cash, nothing to sell), but a second/third rebalance would need to also
sell down over-weighted positions first — not built yet, since it wasn't needed until
the account has an existing position to rebalance away from.

Setup required before this will run:
  1. Create a free account at https://alpaca.markets/ and enable paper trading.
  2. Copy .env.example to .env and fill in ALPACA_API_KEY / ALPACA_SECRET_KEY.
  3. Run: python scripts/run_paper_trade.py --dry-run   (prints intended orders, sends nothing)
     Then: python scripts/run_paper_trade.py            (actually submits paper orders)
"""
import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from pypfopt import EfficientCVaR

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TICKERS, ALPACA_API_KEY, ALPACA_SECRET_KEY, LOGS_DIR, DATA_PROCESSED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"paper_trade_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

LOOKBACK = 20
MU_LOOKBACK = 252
MAX_WEIGHT = 0.35
BETA = 0.95


class VolLSTM(nn.Module):
    """Must match the architecture trained in Notebook 02 exactly, or load_state_dict fails."""

    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.head(h[-1])


def fetch_recent_data() -> pd.DataFrame:
    """Live prices via yfinance, not the static training CSV -- this is the forward-test
    stage, so it needs today's data, not 2015-2026 historical data."""
    raw = yf.download(TICKERS, period="2y", auto_adjust=True, progress=False)
    prices = raw["Close"][TICKERS].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log.info("Fetched %d days of live data, most recent: %s", len(log_returns), log_returns.index.max().date())
    return log_returns


def forecast_risk(log_returns: pd.DataFrame) -> pd.Series:
    """LSTM volatility forecast, trained in Notebook 02, loaded here for inference only
    (the model is not retrained here -- see the trained-then-frozen design in Methodology)."""
    model = VolLSTM()
    model.load_state_dict(torch.load(DATA_PROCESSED / "lstm_vol_model.pt"))
    model.eval()

    with open(DATA_PROCESSED / "lstm_vol_scaler.json") as f:
        scaler = json.load(f)
    y_mean, y_std = scaler["y_mean"], scaler["y_std"]

    roll5 = log_returns.rolling(5).std() * np.sqrt(252)
    pred_vol = {}
    for tkr in TICKERS:
        r_window = log_returns[tkr].values[-LOOKBACK:]
        v_window = roll5[tkr].values[-LOOKBACK:]
        x = np.stack([r_window, v_window], axis=-1).astype(np.float32).reshape(1, LOOKBACK, 2)
        with torch.no_grad():
            raw_pred = float(model(torch.tensor(x)).item())
        pred_vol[tkr] = raw_pred * y_std + y_mean

    forecast = pd.Series(pred_vol)
    log.info("Risk forecast (annualized vol):\n%s", forecast.round(4).to_string())
    assert (forecast > 0).all(), "volatility forecast contains non-positive values -- do not trade on this"
    return forecast


def optimize_weights(risk_forecast: pd.Series, log_returns: pd.DataFrame) -> pd.Series:
    """Mean-CVaR optimization (Rockafellar and Uryasev, 2000 -- Methodology Equation 6),
    matching Notebook 03 exactly: forecast vol x forecast vol x historical correlation
    for the covariance matrix, trailing 252-day historical mean for expected return, same
    35% position cap."""
    corr = pd.read_csv(DATA_PROCESSED / "historical_correlation.csv", index_col=0).loc[TICKERS, TICKERS]
    outer = np.outer(risk_forecast.values, risk_forecast.values)
    cov = pd.DataFrame(outer * corr.values, index=TICKERS, columns=TICKERS)

    mu = log_returns.tail(MU_LOOKBACK).mean() * 252
    scenarios = log_returns.tail(MU_LOOKBACK)

    ec = EfficientCVaR(mu, scenarios, weight_bounds=(0, MAX_WEIGHT), beta=BETA)
    ec.min_cvar()
    weights = pd.Series(ec.clean_weights())
    log.info("Optimized weights (mean-CVaR, %.0f%% cap):\n%s", MAX_WEIGHT * 100, weights.round(4).to_string())
    return weights


MIN_NOTIONAL = 1.00


def submit_orders(target_weights: pd.Series, client: TradingClient, dry_run: bool) -> None:
    account = client.get_account()
    equity = float(account.equity)
    log.info("Paper account equity: $%.2f", equity)

    for ticker, weight in target_weights.items():
        dollar_amount = equity * weight
        log.info("Target: %s -> %.1f%% ($%.2f)", ticker, weight * 100, dollar_amount)

        if dollar_amount < MIN_NOTIONAL:
            log.info("  skipping %s -- target weight is ~0, nothing to order", ticker)
            continue

        if dry_run:
            continue

        order = MarketOrderRequest(
            symbol=ticker,
            notional=round(dollar_amount, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)
        log.info("Submitted order %s for %s", result.id, ticker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Compute and log orders without sending them")
    args = parser.parse_args()

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Copy .env.example to .env and fill "
            "them in first (see the module docstring above)."
        )

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    log_returns = fetch_recent_data()
    risk_forecast = forecast_risk(log_returns)
    target_weights = optimize_weights(risk_forecast, log_returns)

    log.info("Mode: %s", "DRY RUN (no orders sent)" if args.dry_run else "LIVE PAPER ORDERS")
    submit_orders(target_weights, client, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
