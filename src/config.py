"""Shared config: asset universe, paths, and Alpaca credentials.

Import this from notebooks and scripts instead of re-typing the ticker list or
re-reading .env in five different places.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD", "DBC", "VNQ", "HYG"]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
