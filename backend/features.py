import numpy as np
import pandas as pd

def build_features(df, spy, vix):
    df = df.copy()
    df['ret_1d']  = df['Close'].pct_change(1)
    df['ret_5d']  = df['Close'].pct_change(5)
    df['ret_21d'] = df['Close'].pct_change(21)
    df['ret_63d'] = df['Close'].pct_change(63)

    df['vol_10']    = df['ret_1d'].rolling(10).std()
    df['vol_21']    = df['ret_1d'].rolling(21).std()
    df['vol_63']    = df['ret_1d'].rolling(63).std()
    df['vol_ratio'] = df['vol_10'] / df['vol_21'].replace(0, np.nan)

    sma20  = df['Close'].rolling(20).mean()
    sma50  = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    df['price_vs_sma20']  = df['Close'] / sma20 - 1
    df['price_vs_sma50']  = df['Close'] / sma50 - 1
    df['price_vs_sma200'] = df['Close'] / sma200 - 1
    df['sma20_vs_sma50']  = sma20 / sma50 - 1
    df['sma50_vs_sma200'] = sma50 / sma200 - 1

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi14']        = 100 - (100 / (1 + rs))
    df['rsi14_change'] = df['rsi14'].diff()

    ema12  = df['Close'].ewm(span=12).mean()
    ema26  = df['Close'].ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd_norm'] = macd / df['Close']
    df['macd_hist'] = (macd - signal) / df['Close']

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_pos']   = (df['Close'] - bb_mid) / (2 * bb_std).replace(0, np.nan)
    df['bb_width'] = (4 * bb_std) / bb_mid.replace(0, np.nan)

    hl  = df['High'] - df['Low']
    hpc = abs(df['High'] - df['Close'].shift())
    lpc = abs(df['Low']  - df['Close'].shift())
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['atr14_norm'] = tr.rolling(14).mean() / df['Close']
    df['range_pct']  = (df['High'] - df['Low']) / df['Close'].shift()
    df['gap_pct']    = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
    df['close_pos']  = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)

    vs5  = df['Volume'].rolling(5).mean()
    vs21 = df['Volume'].rolling(21).mean()
    df['vol_ratio_5d']  = df['Volume'] / vs5.replace(0, np.nan)
    df['vol_ratio_21d'] = df['Volume'] / vs21.replace(0, np.nan)
    df['obv_ratio']     = (df['Volume'] * np.sign(df['ret_1d'])).rolling(21).sum() / \
                           df['Volume'].rolling(21).sum().replace(0, np.nan)

    spy = spy.copy()
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy_close  = spy['Close']
    spy_sma200 = spy_close.rolling(200).mean()
    spy_ret21  = spy_close.pct_change(21)
    df['spy_above_200ma'] = (spy_close > spy_sma200).astype(int).reindex(df.index).ffill()
    df['spy_ret_21d']     = spy_ret21.reindex(df.index).ffill()

    vix = vix.copy()
    vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix_close = vix['Close']
    vix_sma20 = vix_close.rolling(20).mean()
    df['vix_level']     = vix_close.reindex(df.index).ffill()
    df['vix_5d_change'] = vix_close.pct_change(5).reindex(df.index).ffill()
    df['vix_vs_sma20']  = (vix_close / vix_sma20 - 1).reindex(df.index).ffill()

    df['rs_vs_spy_21d'] = df['ret_21d'] - spy_ret21.reindex(df.index).ffill()
    df['rs_vs_spy_5d']  = df['ret_5d']  - spy_close.pct_change(5).reindex(df.index).ffill()

    return df