"""Definition of constants for use in base module.
"""

_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "n/a",
]
_STORE_INDICATORS_DEFAULT = [
    ("sma", (50,)),
    ("sma", (200,)),
    ("ema", (12,)),
    ("ema", (26,)),
    ("wma", (9,)),
    ("macd", (26, 12, 9)),
    ("adx", (14,)),
    ("rsi", (14,)),
    ("atr", (14,)),
    ("bb", (20, 2)),
    ("cmf", (20,)),
    ("dc", (20, 0)),
    ("kc", (20, 10)),
    ("mfi", (14,)),
    ("obv", ()),
    ("psar", (0.02, 0.2)),
    ("roc", (12,)),
    ("so", (14, 3)),
    ("vwap", (14,)),
    ("dr", ()),
    ("dlr", ()),
]
_STORE_SIGNALS_DEFAULT = [
    ("pairedcross", ("sma_50", "sma_200")),
    ("slopecarry", ("sma_50", 2)),
    ("closecross", ("sma_50",)),
    ("macdcap", (12, 26)),
    ("rsicap", (30, 70)),
    ("adxcap", (25,)),
    ("atrcross", ("sma_50", 0.04)),
    ("bbcross", (0.98,)),
    ("cmfcap", (0.1,)),
    ("dccross", (0.98,)),
    ("kccross", (1.0,)),
    (
        "mficap",
        (
            20,
            80,
            14,
        ),
    ),
    ("psarcross", ()),
    ("roccap", (-5, 5)),
    ("stochcap", (-80, 80, 3)),
    ("vwapcross", (14,)),
]
_TA_COMPUTE_MAP = {
    "obv": [
        "volume",
        "OnBalanceVolumeIndicator",
        ["close", "volume"],
        ["on_balance_volume"],
    ],
    "mfi": [
        "volume",
        "MFIIndicator",
        ["high", "low", "close", "volume"],
        ["money_flow_index"],
    ],
    "cmf": [
        "volume",
        "ChaikinMoneyFlowIndicator",
        ["high", "low", "close", "volume"],
        ["chaikin_money_flow"],
    ],
    "vwap": [
        "volume",
        "VolumeWeightedAveragePrice",
        ["high", "low", "close", "volume"],
        ["volume_weighted_average_price"],
    ],
    "atr": [
        "volatility",
        "AverageTrueRange",
        ["high", "low", "close"],
        ["average_true_range"],
    ],
    "bb": [
        "volatility",
        "BollingerBands",
        ["close"],
        ["bollinger_hband", "bollinger_mavg", "bollinger_lband"],
    ],
    "kc": [
        "volatility",
        "KeltnerChannel",
        ["high", "low", "close"],
        ["keltner_channel_hband", "keltner_channel_mband", "keltner_channel_lband"],
    ],
    "dc": [
        "volatility",
        "DonchianChannel",
        ["high", "low", "close"],
        ["donchian_channel_hband", "donchian_channel_mband", "donchian_channel_lband"],
    ],
    "sma": ["trend", "SMAIndicator", ["close"], ["sma_indicator"]],
    "ema": ["trend", "EMAIndicator", ["close"], ["ema_indicator"]],
    "wma": ["trend", "WMAIndicator", ["close"], ["wma"]],
    "macd": ["trend", "MACD", ["close"], ["macd", "macd_signal", "macd_diff"]],
    "adx": [
        "trend",
        "ADXIndicator",
        ["high", "low", "close"],
        ["adx", "adx_pos", "adx_neg"],
    ],
    "psar": [
        "trend",
        "PSARIndicator",
        ["high", "low", "close"],
        ["psar", "psar_up", "psar_down"],
    ],
    "rsi": ["momentum", "RSIIndicator", ["close"], ["rsi"]],
    "roc": ["momentum", "ROCIndicator", ["close"], ["roc"]],
    "so": [
        "momentum",
        "StochasticOscillator",
        ["close", "high", "low"],
        ["stoch", "stoch_signal"],
    ],
    "dr": ["others", "DailyReturnIndicator", ["close"], ["daily_return"]],
    "dlr": ["others", "DailyLogReturnIndicator", ["close"], ["daily_log_return"]],
}
_TA_NAME_MAP = {
    "bb": ["bb_hband", "bb_mband", "bb_lband"],
    "dc": ["dc_hband", "dc_mband", "dc_lband"],
    "kc": ["kc_hband", "kc_mband", "kc_lband"],
}
