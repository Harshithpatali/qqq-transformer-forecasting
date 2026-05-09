# backend/utils.py

import yfinance as yf
import pandas as pd

from config import TICKER, START_DATE


def download_data():

    df = yf.download(
        TICKER,
        start=START_DATE
    )

    df.reset_index(inplace=True)

    return df