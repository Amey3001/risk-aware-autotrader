
RE-RUNNING WILL NOT REPRODUCE THE NUMBERS IN THE DISSERTATION


Notebook 01 downloads price data from Yahoo Finance up to whatever today's date
is. Every extra trading day shifts the train/test split, which changes the
trained model and everything computed after it. The results in the dissertation
were computed from data running to 20 August 2026, so running the pipeline after
that date will produce different figures.

The outputs saved in each notebook are the ones the dissertation reports.
Re-running overwrites them with values based on a later data window. This is
expected behaviour for a project evaluated on live market data, not an error.


ATTACHED FILES:

01_data_ingestion_eda.ipynb:- Downloads prices, computes log returns, explores the data

02_risk_model.ipynb:- Compares rolling volatility, GARCH(1,1) and the LSTM

03_optimization.ipynb:- Computes weights: equal, Markowitz, and mean-CVaR

04_backtest_evaluation.ipynb:- Simulates portfolio value and scores the three methods

05_explainability.ipynb:- Applies SHAP to the five highest-risk forecasts

run_paper_trade.py:- Places the live orders on Alpaca (use --dry-run to preview)

live_performance.py:- Computes the live performance figures in Table 4.8

Run the notebooks in order. Each one reads files written by the ones before it,
so Notebook 01 has to run first before any of the others will work.


The two scripts need Alpaca paper-trading keys, set as the environment variables
ALPACA_API_KEY and ALPACA_SECRET_KEY, in my case they are placed in a .env file
in the project root.