"""
Objects for unique assets.
"""
import warnings
from datetime import datetime
import time
import ta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from requests.exceptions import ReadTimeout




class _Kline(pd.DataFrame):
    """
    Pandas DataFrame subclass with custom metadata and constructor for Kline class.
    """
    _metadata = ['_asset', '_indicators', '_signals', '_url_scheme', '_root_path', '_store_indicators',
                 '_store_signals', '_cached', '_info']

    @property
    def _constructor(self):
        return _Kline

    @property
    def asset(self):
        """Get asset name.
        """
        return self._asset

    @property
    def name(self):
        """Get asset name.
        """
        return self._asset

    @property
    def indicators(self):
        """Get indicators.
        """
        return self._indicators.reindex(self.index)

    @property
    def signals(self):
        """Get signals.
        """
        return self._signals.reindex(self.index)

    @property
    def url_scheme(self):
        """Get url scheme.
        """
        return self._url_scheme

    @property
    def root_path(self):
        """Get root path.
        """
        return self._root_path

    @property
    def store_indicators(self):
        """Get stored indicators list.
        """
        return self._store_indicators

    @property
    def store_signals(self):
        """Get stored signals list.
        """
        return self._store_signals

    @property
    def cached(self):
        """Get cached indicator.
        """
        return self._cached

    @property
    def info(self):
        """Get asset info.
        """
        return self._info

    @property
    def start(self):
        """Get index start.
        """
        return self.index.min()

    @property
    def end(self):
        """Get index start.
        """
        return self.index.max()

    def _open(self, path, mode):
        if self.url_scheme==str:
            return self.url_scheme(path)
        else:
            return self.url_scheme(path, mode)

    def _get_stored(self):
        """
        Returns:
            (pandas.DataFrame) The asset's stored data.
        """
        try:
            filepath_or_buffer = self._open(self.root_path+self.asset+'.csv', mode='r')
            stored_df = pd.read_csv(filepath_or_buffer=filepath_or_buffer, index_col=0, sep=';', parse_dates=True,
                dtype=np.float64)
        except FileNotFoundError:
            stored_df = None
        return stored_df

    def _get_remote(self, client, start_datetime):
        """
        Args:
            client: Connection client to the crypto exchange.
            start_datetime (datetime): Starting datetime from when to fetch data.
            end_datetime (datetime): Ending datetime to fetch data up to.

        Returns:
            (pandas.DataFrame): Dataframe of the asset's remote data.
        """
        start_timestamp = int(start_datetime.timestamp()*1000)
        klines = client.get_historical_klines(symbol=self.asset, interval=client.KLINE_INTERVAL_1HOUR,
                                              start_str=start_timestamp, # end_str=end_timestamp,
                                              limit=999999999)
        timestamps = [kline[0]/1000 for kline in klines]
        datetimes = [datetime.utcfromtimestamp(timestamp) for timestamp in timestamps]
        remote_df = pd.DataFrame(data=klines, columns=self._cols, index=datetimes, dtype=np.float64)
        return remote_df

    def update(self, client, store=False, verbose=False, sleep=1):
        """
        Args:
            client: Connection client to the crypto exchange.
            store (bool): If True, overwrites the stored data. Else, overwrites only the data in memory.
            verbose (bool): Controls verbosity.
            sleep (int): Sleep interval between update ticks.

        Returns:
            None. Fetches up-to-date data on the crypto exchange, updates self, and stores it if indicated.
        """
        today = datetime.utcnow()
        today_ts = pd.Timestamp(year=today.year, month=today.month, day=today.day, hour=today.hour)
        last_index_max = self.index.max()
        while last_index_max != today_ts:
            start_from_index = (pd.Timestamp(year=2017, month=1, day=1, hour=0, tz='UTC') if pd.isnull(last_index_max)
                                else last_index_max)
            remote_df = self._get_remote(client=client, start_datetime=start_from_index)
            for index in remote_df.index:
                if verbose:
                    print(self.name, index)
                self.loc[index,:] = remote_df.loc[index,:]
            if last_index_max == self.index.max():
                break
            last_index_max = self.index.max()
            time.sleep(sleep)
        if store:
            self.store()
        setattr(self, '_indicators', Indicators(self, store_indicators=self.store_indicators))
        setattr(self, '_signals', Signals(self, store_signals=self.store_signals))

    def store(self):
        """
        Returns:
            None. Overwrites the asset's stored data.
        """
        path_or_buf = self._open(self.root_path+self.asset+'.csv', mode='w')
        self.to_csv(path_or_buf=path_or_buf, sep=';')

    def plot(self, size=180, indicators=None, signals=None, rangeslider=False, secondary=False, fig=None):
        """
        Args:
            size (int): The number of ticks in the past for which to plot the desired graph.
            indicators (list): List of indicators to plot.
            signals (list): List of signals to plot.
            rangeslides (bool): Whether to plot a rangeslider or not.
            secondary (bool or list): Whether or not, or which indicator to plot on a secondary axis.
            fig (plotly.graph_objs.Figure): An existing plotly figure on which to plot.

        Returns:
            None. Plots the assets "open-high-low-close" data in candlestick style,
            overlayed with the desired indicators/signals.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]]) if fig is None else fig
        signals, indicators = [([] if s_or_m is None else s_or_m) for s_or_m in (signals, indicators)]
        colors = ["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e",
                  "#577590", "#277da1"]
        _, y_max = self.low[-size:].min(), self.high[-size:].max()
        fig.add_trace(go.Candlestick(x=self.index[-size:],
                                     open=self.open[-size:],
                                     high=self.high[-size:],
                                     low=self.low[-size:],
                                     close=self.close[-size:],
                                     name=self.name))
        for i, indicator in enumerate(indicators):
            color_index = int(i/len(indicators)*len(colors))
            secondary_y = secondary if isinstance(secondary, bool) else (indicator in secondary)
            fig.add_trace(go.Scatter(x=self.index[-size:],
                                     y=self.indicators.loc[:, indicator][-size:],
                                     line=dict(color=colors[color_index]),
                                     name=indicator,
                                     ), secondary_y=secondary_y)
        for signal in signals:
            signal_data = self.signals.loc[:, signal]
            fig.add_trace(go.Scatter(x=self.index[-size:],
                                     y=np.where(signal_data.eq(1)[-size:], y_max, 0),
                                     mode='none',
                                     fill='tozeroy',
                                     fillcolor='rgba(60, 220, 100, 0.2)',
                                     name=signal+'-BUY'))
            fig.add_trace(go.Scatter(x=self.index[-size:],
                                     y=np.where(signal_data.eq(-1)[-size:], y_max, 0),
                                     mode='none',
                                     fill='tozeroy',
                                     fillcolor='rgba(220, 60, 100, 0.2)',
                                     visible='legendonly',
                                     name=signal+'-SELL'))
        fig.update_layout(xaxis_rangeslider_visible=rangeslider)
        # fig.update_yaxes(range=[y_min, y_max])
        fig.show()

class Kline(_Kline):
    """
    Object for asset-specific data storage, retrieval, and analysis.

    Args:
        asset (str): String of the official base-quote acronym.
        url_scheme (type): Function for the desired url-scheme. Defaults to str.
        root_path (str): Root path of the stored data.
        store_indicators (iterable): List of indicators to store.
        store_signals (iterable): List of signals to store.
        info (pd.Series): Inherited when initialized from KLMngr.
    """
    _cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'n/a']
    _store_indicators_default = [('sma', (50,)),
                                 ('sma', (200,)),
                                 ('ema', (12,)),
                                 ('ema', (26,)),
                                 ('wma', (9,)),
                                 ('macd', (26, 12, 9)),
                                 ('adx', (14,)),
                                 ('rsi', (14,)),
                                 ('atr', (14,)),
                                 ('bb', (20, 2)),
                                 ('cmf', (20,)),
                                 ('dc', (20, 0)),
                                 ('kc', (20, 10)),
                                 ('mfi', (14,)),
                                 ('obv', ()),
                                 ('psar', (0.02, 0.2)),
                                 ('roc', (12,)),
                                 ('so', (14, 3)),
                                 ('vwap', (14,)),
                                 ('dr', ()),
                                 ('dlr', ()),
                                ]
    _store_signals_default = [('crossovers', ('sma_50', 'sma_200')),
                              ('trend', ('sma_50', 2)),
                              ('pricecross', ('sma_50',)),
                              ('macdcross', (12, 26)),
                              ('rsicross', (30, 70)),
                             ]

    def __init__(self, asset, data=None, url_scheme=str, root_path='data/', store_indicators=None, store_signals=None,
                 info=None):
        self._asset = asset
        self._url_scheme = url_scheme
        self._root_path = root_path
        self._store_indicators = self._store_indicators_default if store_indicators is None else store_indicators
        self._store_signals = self._store_signals_default if store_signals is None else store_signals
        self._cached = False
        self._indicators = Indicators(self, store_indicators=self.store_indicators)
        self._signals = Signals(self, store_signals=self.store_signals)
        self._info = info
        if data is not None:
            self._cached=True
            kwargs=dict()
            kwargs['columns'] = self._cols
            super(Kline, self).__init__(data=data, dtype=np.float64, **kwargs)

    def __getattr__(self, attr_name):
        if not self._cached:
            data=self._get_stored()
            self._cached=True
            kwargs=dict()
            kwargs['columns'] = self._cols
            super(Kline, self).__init__(data=data, dtype=np.float64, **kwargs)
        return super(Kline, self).__getattr__(attr_name)

    @property
    def daily(self):
        """Daily resample.
        """
        return self.resample(rule='D')

    @property
    def weekly(self):
        """Weekly resample.
        """
        return self.resample(rule='W')

    @property
    def monthly(self):
        """Monthly resample.
        """
        return self.resample(rule='M')

    def resample(self, rule, axis=0, closed=None, label=None, convention='start', kind=None,
                 loffset=None, base=None, on=None, level=None, origin='start_day', offset=None):
        """Resample.
        """
        agg_dict = {'open_time': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                    'volume': 'sum', 'close_time': 'last', 'quote_asset_volume': 'sum', 'number_of_trades': 'sum',
                    'taker_buy_base_asset_volume': 'sum', 'taker_buy_quote_asset_volume': 'sum', 'n/a': 'mean'}
        data = getattr(super(Kline, self).resample(rule=rule, axis=axis, closed=closed, label=label, on=on,
                                                   convention=convention, kind=kind, loffset=loffset, base=base,
                                                   level=level, origin=origin, offset=offset), 'agg')(agg_dict)
        return Kline(asset=self.asset, data=data, url_scheme=self._url_scheme, root_path=self._root_path,
                     store_indicators=self._store_indicators, store_signals=self._store_signals, info=self._info)


class _Indicators(pd.DataFrame):
    """
    Pandas DataFrame subclass with custom metadata and constructor for Indicators class.
    """
    _metadata = ['_kline', '_cached', '_store_indicators']
    _ta_compute_map = {
        'obv': ['volume', 'OnBalanceVolumeIndicator', ['close', 'volume'], ['on_balance_volume']],
        'mfi': ['volume', 'MFIIndicator', ['high', 'low', 'close', 'volume'], ['money_flow_index']],
        'cmf': ['volume', 'ChaikinMoneyFlowIndicator', ['high', 'low', 'close', 'volume'], ['chaikin_money_flow']],
        'vwap': ['volume', 'VolumeWeightedAveragePrice', ['high', 'low', 'close', 'volume'],
                 ['volume_weighted_average_price']],
        'atr': ['volatility', 'AverageTrueRange', ['high', 'low', 'close'], ['average_true_range']],
        'bb': ['volatility', 'BollingerBands', ['close'], ['bollinger_hband', 'bollinger_mavg', 'bollinger_lband']],
        'kc': ['volatility', 'KeltnerChannel', ['high', 'low', 'close'], ['keltner_channel_hband',
               'keltner_channel_mband', 'keltner_channel_lband']],
        'dc': ['volatility', 'DonchianChannel', ['high', 'low', 'close'], ['donchian_channel_hband',
               'donchian_channel_mband', 'donchian_channel_lband']],
        'sma': ['trend', 'SMAIndicator', ['close'], ['sma_indicator']],
        'ema': ['trend', 'EMAIndicator', ['close'], ['ema_indicator']],
        'wma': ['trend', 'WMAIndicator', ['close'], ['wma']],
        'macd': ['trend', 'MACD', ['close'], ['macd', 'macd_signal', 'macd_diff']],
        'adx': ['trend', 'ADXIndicator', ['high', 'low', 'close'], ['adx', 'adx_pos', 'adx_neg']],
        'psar': ['trend', 'PSARIndicator', ['high', 'low', 'close'], ['psar', 'psar_up', 'psar_down']],
        'rsi': ['momentum', 'RSIIndicator', ['close'], ['rsi']],
        'roc': ['momentum', 'ROCIndicator', ['close'], ['roc']],
        'so': ['momentum', 'StochasticOscillator', ['close', 'high', 'low'], ['stoch', 'stoch_signal']],
        'dr': ['others', 'DailyReturnIndicator', ['close'], ['daily_return']],
        'dlr': ['others', 'DailyLogReturnIndicator', ['close'], ['daily_log_return']],
    }
    _ta_name_map = {
        'bb': ['bb_hband', 'bb_mband', 'bb_lband'],
        'dc': ['dc_hband', 'dc_mband', 'dc_lband'],
        'kc': ['kc_hband', 'kc_mband', 'kc_lband'],
    }

    @property
    def _constructor(self):
        """Requirement for correct pd.DataFrame subclassing.
        """
        return _Indicators

    @property
    def kline(self):
        """Get underlying kline.
        """
        return self._kline

    @property
    def cached(self):
        """Get cached indicator.
        """
        return self._cached

    @property
    def store_indicators(self):
        """Get stored indicators list.
        """
        return self._store_indicators

    def compute(self, indicator, *args, **kwargs):
        """Compute an indicator.
        """
        module_name, class_name, value_names, method_names = self._ta_compute_map[indicator]
        default_args = (getattr(self.kline, value_name) for value_name in value_names)
        ta_class = getattr(getattr(ta, module_name), class_name)(*default_args, *args, **kwargs)
        computed_indicators = []
        for method_name in method_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                computed_indicators.append(getattr(ta_class, method_name)())
        df_indicators = pd.concat(computed_indicators, axis=1)
        if indicator in self._ta_name_map:
            df_indicators.columns = self._ta_name_map[indicator]
        return df_indicators

    def append(self, computed_indicator):
        """Append an indicator.
        """
        for column in computed_indicator:
            self.loc[:, column.lower()] = computed_indicator[column]
        return None

    def extend(self, indicator, *args, **kwargs):
        """Compute and append an indicator.
        """
        computed_indicator = self.compute(indicator, *args, **kwargs)
        self.append(computed_indicator)
        return None


class Indicators(_Indicators):
    """
    Object for asset-specific indicators manipulations.

    Args:
        kline (Kline): Kline from which to get info to compute indicators.
        store_indicators (list): Indicator names to store in memory.
    """
    def __init__(self, kline, store_indicators):
        self._kline=kline
        self._cached=False
        self._store_indicators=store_indicators
        # super().__init__()

    def __getattr__(self, attr_name):
        if not self._cached:
            self._cached=True
            data=pd.DataFrame(data=[], index=self.kline.index)
            super(Indicators, self).__init__(data=data)
            for indicator in self.store_indicators:
                try:
                    fun, args = indicator
                except ValueError:
                    fun = indicator
                self.extend(fun, *args)
        return super(Indicators, self).__getattr__(attr_name)


class _Signals(pd.DataFrame):
    """
    Pandas DataFrame subclass with custom metadata and constructor for Signals class.
    """
    _metadata = ['_kline', '_cached', '_store_signals']

    @property
    def _constructor(self):
        return _Signals

    @property
    def kline(self):
        """Get underlying kline.
        """
        return self._kline

    @property
    def cached(self):
        """Get cached indicator.
        """
        return self._cached

    @property
    def store_signals(self):
        """Get stored signals list.
        """
        return self._store_signals

    def compute(self, signal, *args, **kwargs):
        """Compute a signal.
        """
        fun = signal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            computed_signal = getattr(self, '_compute_signal_'+fun)(self.kline, *args, **kwargs)
        return computed_signal

    def append(self, computed_signal, signal, *args, **kwargs):
        """Append a signal.
        """
        str_args = '_' + '_'.join([str(_) for _ in args]) if len(args) > 0 else ''
        str_kwargs = '_' + '_'.join([str(kwargs[_]) for _ in kwargs]) if len(kwargs) > 0 else ''
        signal_name = signal + str_args + str_kwargs
        self.loc[:, signal_name] = computed_signal
        return None

    def extend(self, signal, *args, **kwargs):
        """Compute and append a signal.
        """
        computed_signal = self.compute(signal, *args, **kwargs)
        self.append(computed_signal, signal, *args, **kwargs)
        return None

    @staticmethod
    def _compute_signal_crossovers(kline, fast_indicator='sma_50', slow_indicator='sma_200', buffer=.0001):
        sl_diff = (getattr(kline.indicators, fast_indicator) -
                   getattr(kline.indicators, slow_indicator))/getattr(kline.indicators, fast_indicator)
        # crossovers = ((sl_diff > 0) & (sl_diff > sl_diff.shift(1))).map({True: 1., False: -1.})
        bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        cuts = pd.cut(sl_diff, bins=bins, duplicates='drop', labels=False)-1.
        return cuts.where(cuts != 1, cuts.where(sl_diff.diff(1) > 0, 0))

    @staticmethod
    def _compute_signal_trend(kline, indicator='sma_50', repeat=2, buffer=.0001):
        ydiff = getattr(kline.indicators, indicator).pct_change(1)
        index_diff = kline.index.to_series().diff(1)
        seconds_per_step = index_diff.value_counts().index[0].total_seconds()
        xdiff = (index_diff.dt.total_seconds()/seconds_per_step).replace(0, np.nan)
        slopes = (ydiff/xdiff)
        bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        slopes_bins = pd.cut(slopes, bins=bins, duplicates='drop', labels=False)-1
        slopes_bins_rolling = slopes_bins.rolling(repeat).sum()
        mask = (slopes_bins_rolling <= -repeat) | (slopes_bins_rolling >= repeat)
        return slopes_bins_rolling.where(mask).fillna(0).clip(-1., 1.)

    @staticmethod
    def _compute_signal_pricecross(kline, indicator='sma_50', buffer=.0001):
        price_indicator_diff = (kline.close - getattr(kline.indicators, indicator))/kline.close
        bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        return pd.cut(price_indicator_diff, bins=bins, duplicates='drop', labels=False)-1.

    @staticmethod
    def _compute_signal_macdcross(kline, fast_window=12, slow_window=26, buffer=.0001):
        macd = getattr(kline.indicators, 'macd_{}_{}'.format(fast_window, slow_window))
        macd_diff = getattr(kline.indicators, 'macd_diff_{}_{}'.format(fast_window, slow_window))
        macd_signal_diff = macd_diff/macd
        bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        return pd.cut(macd_signal_diff, bins=bins, duplicates='drop', labels=False)-1.

    @staticmethod
    def _compute_signal_rsicross(kline, lower=30, upper=70): # window=14
        rsi = getattr(kline.indicators, 'rsi')
        return -(pd.cut(rsi, bins=[-float('inf'), lower, upper, float('inf')], labels=False)-1.)


class Signals(_Signals):
    """
    Object for asset-specific signals manipulations.

    Args:
        kline (Kline): Kline from which to get info to compute signals.
        store_signals (list): Signal names to store in memory.
    """
    def __init__(self, kline, store_signals):
        self._kline=kline
        self._cached=False
        self._store_signals = store_signals
        # super().__init__()

    def __getattr__(self, attr_name):
        if not self._cached:
            self._cached=True
            data=pd.DataFrame(data=[], index=self.kline.index)
            super(Signals, self).__init__(data=data)
            for signal in self.store_signals:
                try:
                    fun, args = signal
                except ValueError:
                    fun = signal
                self.extend(fun, *args)
        return super(Signals, self).__getattr__(attr_name)
