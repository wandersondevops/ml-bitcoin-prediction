# Projeto 2 - Construção e Deploy de API - Machine Learning Para Prever o Preço do Bitcoin
# Módulo de Indicadores

# Imports
import numpy as np
import pandas as pd

# Função de cálculo dos indicadores
def dsa_calcula_indicadores(dsa_dados) :

    dsa_dados = dsa_ind_williams_percent_r(dsa_dados,14)
    dsa_dados = dsa_ind_roc(dsa_dados,14)
    dsa_dados = dsa_ind_rsi(dsa_dados,7)
    dsa_dados = dsa_ind_rsi(dsa_dados,14)
    dsa_dados = dsa_ind_rsi(dsa_dados,28)
    dsa_dados = dsa_ind_macd(dsa_dados, 8, 21)
    dsa_dados = dsa_ind_bbands(dsa_dados,20)    
    dsa_dados = dsa_ind_ichimoku_cloud(dsa_dados)
    dsa_dados = dsa_ind_ema(dsa_dados, 3)
    dsa_dados = dsa_ind_ema(dsa_dados, 8)
    dsa_dados = dsa_ind_ema(dsa_dados, 15)
    dsa_dados = dsa_ind_ema(dsa_dados, 50)
    dsa_dados = dsa_ind_ema(dsa_dados, 100)
    dsa_dados = dsa_ind_adx(dsa_dados, 14)
    dsa_dados = dsa_ind_donchian(dsa_dados, 10)
    dsa_dados = dsa_ind_donchian(dsa_dados, 20)
    dsa_dados = dsa_ind_alma(dsa_dados, 10)
    dsa_dados = dsa_ind_tsi(dsa_dados, 13, 25)
    dsa_dados = dsa_ind_zscore(dsa_dados, 20)
    dsa_dados = dsa_ind_log_return(dsa_dados, 10)
    dsa_dados = dsa_ind_log_return(dsa_dados, 20)
    dsa_dados = dsa_ind_vortex(dsa_dados, 7)
    dsa_dados = dsa_ind_aroon(dsa_dados, 16)
    dsa_dados = dsa_ind_ebsw(dsa_dados, 14)
    dsa_dados = dsa_ind_accbands(dsa_dados, 20)
    dsa_dados = dsa_ind_short_run(dsa_dados, 14)
    dsa_dados = dsa_ind_bias(dsa_dados, 26)
    dsa_dados = dsa_ind_ttm_trend(dsa_dados, 5, 20)
    dsa_dados = dsa_ind_percent_return(dsa_dados, 10)
    dsa_dados = dsa_ind_percent_return(dsa_dados, 20)
    dsa_dados = dsa_ind_kurtosis(dsa_dados, 5)
    dsa_dados = dsa_ind_kurtosis(dsa_dados, 10)
    dsa_dados = dsa_ind_kurtosis(dsa_dados, 20)
    dsa_dados = dsa_ind_eri(dsa_dados, 13)    
    dsa_dados = dsa_ind_atr(dsa_dados, 14)
    dsa_dados = dsa_ind_keltner_channels(dsa_dados, 20)
    dsa_dados = dsa_ind_chaikin_volatility(dsa_dados, 10)
    dsa_dados = dsa_ind_stdev(dsa_dados, 5)
    dsa_dados = dsa_ind_stdev(dsa_dados, 10)
    dsa_dados = dsa_ind_stdev(dsa_dados, 20)
    dsa_dados = ta_vix(dsa_dados, 21)    
    dsa_dados = dsa_ind_obv(dsa_dados, 10)
    dsa_dados = dsa_ind_chaikin_money_flow(dsa_dados, 5)
    dsa_dados = dsa_ind_volume_price_trend(dsa_dados, 7)
    dsa_dados = dsa_ind_accumulation_distribution_line(dsa_dados, 3)
    dsa_dados = dsa_ind_ease_of_movement(dsa_dados, 14)
    
    return dsa_dados

# Williams %R
def dsa_ind_williams_percent_r(dados_dsa, window=14):
    highest_high = dados_dsa["High"].rolling(window=window).max()
    lowest_low = dados_dsa["Low"].rolling(window=window).min()
    dados_dsa["Williams_%R{}".format(window)] = -((highest_high - dados_dsa["Close"]) / (highest_high - lowest_low)) * 100
    return dados_dsa

# Rate of Change
def dsa_ind_roc(dados_dsa, window=14):
    dados_dsa["ROC_{}".format(window)] = (dados_dsa["Close"] / dados_dsa["Close"].shift(window) - 1) * 100
    return dados_dsa

# RSI
def dsa_ind_rsi(dados_dsa, window=14) : 
    delta = dados_dsa["Close"].diff(1)
    gains = delta.where(delta>0,0)
    losses = -delta.where(delta<0,0)
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    dados_dsa["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return dados_dsa

# MACD 
def dsa_ind_macd(dados_dsa, short_window=8, long_window=21, signal_window=9):
    short_ema = dados_dsa["Close"].ewm(span = short_window, adjust = False).mean()
    long_ema = dados_dsa["Close"].ewm(span = long_window, adjust = False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    dados_dsa["MACD_Line"] = macd_line
    dados_dsa["Signal_Line"] = signal_line
    dados_dsa["MACD_Histogram"] = macd_histogram
    return dados_dsa

# Bollinger Bands
def dsa_ind_bbands(dados_dsa, window=20, num_std_dev=2) :
    dados_dsa["midlle_band"] = dados_dsa["Close"].rolling(window=window).mean()
    dados_dsa["std"] = dados_dsa["Close"].rolling(window=window).std()
    dados_dsa["upper_band{}".format(window)] = dados_dsa["midlle_band"] + (num_std_dev * dados_dsa["std"])
    dados_dsa["lower_band{}".format(window)] = dados_dsa["midlle_band"] - (num_std_dev * dados_dsa["std"])
    dados_dsa.drop(["std"], axis=1, inplace=True)   
    return dados_dsa

# Ichimoku Cloud
def dsa_ind_ichimoku_cloud(dados_dsa, window_tenkan=9, window_kijun=26, window_senkou_span_b=52, window_chikou=26):
    tenkan_sen = (dados_dsa["Close"].rolling(window=window_tenkan).max() + dados_dsa["Close"].rolling(window=window_tenkan).min()) / 2
    kijun_sen = (dados_dsa["Close"].rolling(window=window_kijun).max() + dados_dsa["Close"].rolling(window=window_kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    senkou_span_b = (dados_dsa["Close"].rolling(window=window_senkou_span_b).max() + dados_dsa["Close"].rolling(window=window_senkou_span_b).min()) / 2
    chikou_span = dados_dsa["Close"].shift(-window_chikou)
    dados_dsa["Tenkan_sen"] = tenkan_sen
    dados_dsa["Kijun_sen"] = kijun_sen
    dados_dsa["Senkou_Span_A"] = senkou_span_a
    dados_dsa["Senkou_Span_B"] = senkou_span_b
    dados_dsa["Chikou_Span"] = chikou_span
    return dados_dsa

# Moving Average (EMA)
def dsa_ind_ema(dados_dsa, window=8): 
    dados_dsa["ema_{}".format(window)] = dados_dsa["Close"].ewm(span=window, adjust=False).mean()
    return dados_dsa

# ADX
def dsa_ind_adx(dados_dsa, window=14): #14
    dados_dsa["TR"] = abs(dados_dsa["High"] - dados_dsa["Low"]).combine_first(abs(dados_dsa["High"] - dados_dsa["Close"].shift(1))).combine_first(abs(dados_dsa["Low"] - dados_dsa["Close"].shift(1)))
    dados_dsa["DMplus"] = (dados_dsa["High"] - dados_dsa["High"].shift(1)).apply(lambda x: x if x > 0 else 0)
    dados_dsa["DMminus"] = (dados_dsa["Low"].shift(1) - dados_dsa["Low"]).apply(lambda x: x if x > 0 else 0)
    dados_dsa["ATR"] = dados_dsa["TR"].rolling(window=window).mean()
    dados_dsa["DIplus"] = (dados_dsa["DMplus"].rolling(window=window).mean() / dados_dsa["ATR"]) * 100
    dados_dsa["DIminus"] = (dados_dsa["DMminus"].rolling(window=window).mean() / dados_dsa["ATR"]) * 100
    dados_dsa["DX"] = abs(dados_dsa["DIplus"] - dados_dsa["DIminus"]) / (dados_dsa["DIplus"] + dados_dsa["DIminus"]) * 100
    dados_dsa["ADX_{}".format(window)] = dados_dsa["DX"].rolling(window=window).mean()
    dados_dsa.drop(["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"], axis=1, inplace=True)
    return dados_dsa

# Donchian Channel
def dsa_ind_donchian(dados_dsa, window=10):
    highest_high = dados_dsa["Close"].rolling(window=window).max()
    lowest_low = dados_dsa["Close"].rolling(window=window).min()
    dados_dsa["Donchian_Upper_{}".format(window)] = highest_high
    dados_dsa["Donchian_Lower_{}".format(window)] = lowest_low
    return dados_dsa

# Arnaud Legoux Moving Average (ALMA)
def dsa_ind_alma(dados_dsa, window=10, sigma=6, offset=0.85):
    m = np.linspace(-offset*(window-1), offset*(window-1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    alma_values = np.convolve(dados_dsa["Close"].values, w, mode="valid")
    alma_values = np.concatenate([np.full(window-1, np.nan), alma_values])
    dados_dsa["ALMA_{}".format(window)] = alma_values
    return dados_dsa

# True Strength Index (TSI)
def dsa_ind_tsi(dados_dsa, short_period=13, long_period=25):
    price_diff = dados_dsa["Close"].diff(1)
    double_smoothed = price_diff.ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    double_smoothed_abs = price_diff.abs().ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    dados_dsa["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return dados_dsa

# Z-Score
def dsa_ind_zscore(dados_dsa, window=20):
    rolling_mean = dados_dsa["Close"].rolling(window=window).mean()
    rolling_std = dados_dsa["Close"].rolling(window=window).std()
    z_score = (dados_dsa["Close"] - rolling_mean) / rolling_std
    dados_dsa["Z_Score_{}".format(window)] = z_score
    return dados_dsa

# Log Return
def dsa_ind_log_return(dados_dsa, window=5):
    dados_dsa["LogReturn_{}".format(window)] = dados_dsa["Close"].pct_change(window).apply(lambda x: 0 if pd.isna(x) else x)
    return dados_dsa

# Vortex Indicator
def dsa_ind_vortex(dados_dsa, window=7): 
    high_low = dados_dsa["High"] - dados_dsa["Low"]
    high_close_previous = abs(dados_dsa["High"] - dados_dsa["Close"].shift(1))
    low_close_previous = abs(dados_dsa["Low"] - dados_dsa["Close"].shift(1))
    true_range = pd.concat([high_low, high_close_previous, low_close_previous], axis=1).max(axis=1)
    positive_vm = abs(dados_dsa["High"].shift(1) - dados_dsa["Low"])
    negative_vm = abs(dados_dsa["Low"].shift(1) - dados_dsa["High"])
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    dados_dsa["Positive_VI_{}".format(window)] = positive_vi
    dados_dsa["Negative_VI_{}".format(window)] = negative_vi
    return dados_dsa

# Aroon Indicator
def dsa_ind_aroon(dados_dsa, window=16):
    high_prices = dados_dsa["High"]
    low_prices = dados_dsa["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window:i + 1]
        low_period = low_prices[i - window:i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down
    dados_dsa["Aroon_Up_{}".format(window)] = aroon_up
    dados_dsa["Aroon_Down_{}".format(window)] = aroon_down
    return dados_dsa

# Elder"s Bull Power e Bear Power 
def dsa_ind_ebsw(dados_dsa, window=14):
    ema = dados_dsa["Close"].ewm(span=window, adjust=False).mean()
    bull_power = dados_dsa["High"] - ema
    bear_power = dados_dsa["Low"] - ema
    dados_dsa["Bull_Power_{}".format(window)] = bull_power
    dados_dsa["Bear_Power_{}".format(window)] = bear_power
    return dados_dsa

# Acceleration Bands
def dsa_ind_accbands(dados_dsa, window=20, acceleration_factor=0.02):
    sma = dados_dsa["Close"].rolling(window=window).mean()
    band_difference = dados_dsa["Close"] * acceleration_factor
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    dados_dsa["Upper_Band_{}".format(window)] = upper_band
    dados_dsa["Lower_Band_{}".format(window)] = lower_band
    dados_dsa["Middle_Band_{}".format(window)] = sma
    return dados_dsa

# Short Run
def dsa_ind_short_run(dados_dsa, window=14):
    short_run = dados_dsa["Close"] - dados_dsa["Close"].rolling(window=window).min()
    dados_dsa["Short_Run_{}".format(window)] = short_run
    return dados_dsa

# Bias
def dsa_ind_bias(dados_dsa, window=26):
    moving_average = dados_dsa["Close"].rolling(window=window).mean()
    bias = ((dados_dsa["Close"] - moving_average) / moving_average) * 100
    dados_dsa["Bias_{}".format(window)] = bias
    return dados_dsa

# TTM Trend
def dsa_ind_ttm_trend(dados_dsa, short_window=5, long_window=20):
    short_ema = dados_dsa["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dados_dsa["Close"].ewm(span=long_window, adjust=False).mean()
    ttm_trend = short_ema - long_ema
    dados_dsa["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return dados_dsa

# Percent Return
def dsa_ind_percent_return(dados_dsa, window=1): 
    percent_return = dados_dsa["Close"].pct_change().rolling(window=window).mean() * 100
    dados_dsa["Percent_Return_{}".format(window)] = percent_return
    return dados_dsa

# Kurtosis
def dsa_ind_kurtosis(dados_dsa, window=20):
    dados_dsa["kurtosis_{}".format(window)] = dados_dsa["Close"].rolling(window=window).apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    return dados_dsa

# Elder's Force Index (ERI)
def dsa_ind_eri(dados_dsa, window=13):
    price_change = dados_dsa["Close"].diff()
    force_index = price_change * dados_dsa["Volume"]
    eri = force_index.ewm(span=window, adjust=False).mean()
    dados_dsa["ERI_{}".format(window)] = eri
    return dados_dsa

# ATR
def dsa_ind_atr(dados_dsa, window=14):
    dados_dsa["High-Low"] = dados_dsa["High"] - dados_dsa["Low"]
    dados_dsa["High-PrevClose"] = abs(dados_dsa["High"] - dados_dsa["Close"].shift(1))
    dados_dsa["Low-PrevClose"] = abs(dados_dsa["Low"] - dados_dsa["Close"].shift(1))
    dados_dsa["TrueRange"] = dados_dsa[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    dados_dsa["atr_{}".format(window)] = dados_dsa["TrueRange"].rolling(window=window, min_periods=1).mean()
    dados_dsa.drop(["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"], axis=1, inplace=True)
    return dados_dsa

# Keltner Channels
def dsa_ind_keltner_channels(dados_dsa, period=20, multiplier=2):
    dados_dsa["TR"] = dados_dsa.apply(lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1)
    dados_dsa["ATR"] = dados_dsa["TR"].rolling(window=period).mean()
    dados_dsa["Middle Band"] = dados_dsa["Close"].rolling(window=period).mean()
    dados_dsa["Upper Band"] = dados_dsa["Middle Band"] + multiplier * dados_dsa["ATR"]
    dados_dsa["Lower Band"] = dados_dsa["Middle Band"] - multiplier * dados_dsa["ATR"]
    return dados_dsa

# Chaikin Volatility
def dsa_ind_chaikin_volatility(dados_dsa, window=10):
    daily_returns = dados_dsa["Close"].pct_change()
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252 ** 0.5)
    dados_dsa["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return dados_dsa

# Standard Deviation 
def dsa_ind_stdev(dados_dsa, window=1): 
    stdev_column = dados_dsa["Close"].rolling(window=window).std()
    dados_dsa["Stdev_{}".format(window)] = stdev_column
    return dados_dsa

# Volatility Index (VIX)
def ta_vix(dados_dsa, window=21):
    returns = dados_dsa["Close"].pct_change().dropna()
    rolling_std = returns.rolling(window=window).std()
    vix = rolling_std * np.sqrt(252) * 100  
    dados_dsa["VIX_{}".format(window)] = vix
    return dados_dsa

# On-Balance Volume (OBV)
def dsa_ind_obv(dados_dsa, window=10):
    price_changes = dados_dsa["Close"].diff()
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    obv = (dados_dsa["Volume"] * volume_direction).cumsum()
    obv_smoothed = obv.rolling(window=window).mean()
    dados_dsa["OBV_{}".format(window)] = obv_smoothed
    return dados_dsa

# Chaikin Money Flow (CMF)
def dsa_ind_chaikin_money_flow(dados_dsa, window=10):
    mf_multiplier = ((dados_dsa["Close"] - dados_dsa["Close"].shift(1)) + (dados_dsa["Close"] - dados_dsa["Close"].shift(1)).abs()) / 2
    mf_volume = mf_multiplier * dados_dsa["Volume"]
    adl = mf_volume.cumsum()
    cmf = adl.rolling(window=window).mean() / dados_dsa["Volume"].rolling(window=window).mean()
    dados_dsa["CMF_{}".format(window)] = cmf
    return dados_dsa

# Volume Price Trend (VPT)
def dsa_ind_volume_price_trend(dados_dsa, window=10):
    price_change = dados_dsa["Close"].pct_change()
    vpt = (price_change * dados_dsa["Volume"].shift(window)).cumsum()
    dados_dsa["VPT_{}".format(window)] = vpt
    return dados_dsa

# Accumulation/Distribution Line
def dsa_ind_accumulation_distribution_line(dados_dsa, window=10):
    money_flow_multiplier = ((dados_dsa["Close"] - dados_dsa["Close"].shift(1)) - (dados_dsa["Close"].shift(1) - dados_dsa["Close"])) / (dados_dsa["Close"].shift(1) - dados_dsa["Close"])
    money_flow_volume = money_flow_multiplier * dados_dsa["Volume"]
    ad_line = money_flow_volume.cumsum()
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    dados_dsa["A/D Line_{}".format(window)] = ad_line_smoothed
    return dados_dsa

# Ease of Movement (EOM)
def dsa_ind_ease_of_movement(dados_dsa, window=14):
    midpoint_move = ((dados_dsa["High"] + dados_dsa["Low"]) / 2).diff(1)
    box_ratio = dados_dsa["Volume"] / 1000000 / (dados_dsa["High"] - dados_dsa["Low"])
    eom = midpoint_move / box_ratio
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    dados_dsa["EOM_{}".format(window)] = eom_smoothed
    return dados_dsa
    

