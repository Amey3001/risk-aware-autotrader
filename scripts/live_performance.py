"""Compute the live paper-trading performance metrics reported in Chapter 4.

Pulls the account's daily equity curve from Alpaca's portfolio-history endpoint and
computes total return, maximum drawdown, best and worst day, and the number of trading
days. The window defaults to the period the strategy was actually live: the first orders
were placed on 31 July 2026, so anything before that is the account sitting in cash and
is excluded.

Run: python scripts/live_performance.py
     python scripts/live_performance.py --start 2026-07-31 --end 2026-08-19
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_PROCESSED

BASE_URL = "https://paper-api.alpaca.markets"


def fetch_equity_curve(start, end):
    """Daily account equity between two dates, as a pandas Series."""
    resp = requests.get(
        f"{BASE_URL}/v2/account/portfolio/history",
        headers={
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        },
        params={"start": start, "end": end, "timeframe": "1D"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    equity = pd.Series(
        data["equity"],
        index=pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date,
        dtype=float,
    )
    return equity[equity > 0].dropna()


def compute_live_metrics(equity):
    daily_returns = equity.pct_change().dropna()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "max_drawdown": drawdown.min(),
        "best_day": daily_returns.max(),
        "worst_day": daily_returns.min(),
        "trading_days": len(daily_returns),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-07-31", help="first day the strategy held positions")
    p.add_argument("--end", default="2026-08-20", help="exclusive upper bound on the window")
    args = p.parse_args()

    equity = fetch_equity_curve(args.start, args.end)
    metrics = compute_live_metrics(equity)

    print(f"Live paper-trading performance, {equity.index[1]} to {equity.index[-1]}\n")
    print(f"  Total return   {metrics['total_return']:+.4%}")
    print(f"  Max drawdown   {metrics['max_drawdown']:+.4%}")
    print(f"  Best day       {metrics['best_day']:+.4%}")
    print(f"  Worst day      {metrics['worst_day']:+.4%}")
    print(f"  Trading days   {metrics['trading_days']}")

    out = DATA_PROCESSED / "live_performance.csv"
    pd.Series(metrics).to_csv(out, header=["value"])
    equity.to_csv(DATA_PROCESSED / "live_equity_curve.csv", header=["equity"])
    print(f"\nSaved: {out.name}, live_equity_curve.csv")


if __name__ == "__main__":
    main()
