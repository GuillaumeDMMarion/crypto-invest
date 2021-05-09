"""
Objects for unique assets.
"""
from datetime import datetime
import time
import ta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# from requests.exceptions import ReadTimeout




class _Kline(pd.DataFrame):
    """
    Pandas DataFrame subclass with custom metadata and constructor for Kline class.
    """
    _metadata = ['_asset', '_indicators', '_metrics', '_signals', '_url_scheme', '_root_path', '_store_indicators',
                 '_store_metrics', '_store_signals', '_cached', '_info']

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
    def metrics(self):
        """Get metrics.
        """
        return self._metrics.reindex(self.index)

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
    def store_metrics(self):
        """Get stored metrics list.
        """
        return self._store_metrics

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
        setattr(self, '_metrics', Metrics(self, store_metrics=self.store_metrics))
        setattr(self, '_signals', Signals(self, store_signals=self.store_signals))

    def store(self):
        """
        Returns:
            None. Overwrites the asset's stored data.
        """
        path_or_buf = self._open(self.root_path+self.asset+'.csv', mode='w')
        self.to_csv(path_or_buf=path_or_buf, sep=';')

    def plot(self, size=180, metrics=None, signals=None, rangeslider=False, fig=None):
        """
        Args:
            size (int): The number of ticks in the past for which to plot the desired graph.
            metrics (list): List of metrics to plot.
            signals (list): List of signals to plot.
            rangeslides (bool): Whether to plot a rangeslider or not.
            fig (plotly.graph_objs.Figure): An exising plotly figure on which to plot.

        Returns:
            None. Plots the assets "open-high-low-close" data in candlestick style,
            overlayed with the desired metrics/signals.
        """
        fig = go.Figure() if fig is None else fig
        signals, metrics = [([] if s_or_m is None else s_or_m) for s_or_m in (signals, metrics)]
        colors = ["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e",
                  "#577590", "#277da1"]
        y_min, y_max = self.low[-size:].min(), self.high[-size:].max()
        fig.add_trace(go.Candlestick(x=self.index[-size:],
                                     open=self.open[-size:],
                                     high=self.high[-size:],
                                     low=self.low[-size:],
                                     close=self.close[-size:],
                                     name=self.name))
        for i, metric in enumerate(metrics):
            color_index = int(i/len(metrics)*len(colors))
            fig.add_trace(go.Scatter(x=self.index[-size:],
                                     y=self.metrics.loc[:, metric][-size:],
                                     line=dict(color=colors[color_index]),
                                     name=metric))
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
        fig.update_yaxes(range=[y_min, y_max])
        fig.show()

class Kline(_Kline):
    """
    Object for asset-specific data storage, retrieval, and analysis.

    Args:
        asset (str): String of the official base-quote acronym.
        url_scheme (type): Function for the desired url-scheme. Defaults to str.
        root_path (str): Root path of the stored data.
        store_metrics (iterable): List of metrics to store.
        store_signals (iterable): List of signals to store.
        info (pd.Series): Inherited when initialized from KLMngr.
    """
    _cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'n/a']
    _store_indicators_default = [('sma', (50,)),
                                 ('sma', (200,)),
                                 ('ema', (12,)),
                                 ('ema', (26,)),
                                 ('macd', (26, 12, 9)),
                                 ('adx', (14,))]
    _store_metrics_default = [('SMA', (50,)),
                              ('SMA', (200,)),
                              ('EMA', (12,)),
                              ('EMA', (26, ))]
    _store_signals_default = [('CROSSOVERS', ('SMA_50', 'SMA_200')),
                              ('TREND', ('SMA_50', 2)),
                              ('PRICECROSS', ('SMA_50', )),
                              ('MACDCROSS', ('EMA_12', 'EMA_26', 'EMA', 9)),
                              ('RSICROSS', (14, 30, 70))]

    def __init__(self, asset, data=None, url_scheme=str, root_path='data/', store_indicators=None, store_metrics=None,
                 store_signals=None, info=None):
        self._asset = asset
        self._url_scheme = url_scheme
        self._root_path = root_path
        self._store_indicators = self._store_indicators_default if store_indicators is None else store_indicators
        self._store_metrics = self._store_metrics_default if store_metrics is None else store_metrics
        self._store_signals = self._store_signals_default if store_signals is None else store_signals
        self._cached = False
        self._indicators = Indicators(self, store_indicators=self.store_indicators)
        self._metrics = Metrics(self, store_metrics=self.store_metrics)
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
                     store_metrics=self._store_metrics, store_signals=self._store_signals, info=self._info)


class _Indicators(pd.DataFrame):
    """
    Pandas DataFrame subclass with custom metadata and constructor for Indicators class.
    """
    _metadata = ['_kline', '_cached', '_store_indicators']
    _ta_map = {
        'sma': ['trend', 'SMAIndicator', ['close'], ['sma_indicator']],
        'ema': ['trend', 'EMAIndicator', ['close'], ['ema_indicator']],
        'wma': ['trend', 'WMAIndicator', ['close'], ['wma']],
        'macd': ['trend', 'MACD', ['close'], ['macd', 'macd_signal', 'macd_diff']],
        'adx': ['trend', 'ADXIndicator', ['high', 'low', 'close'], ['adx', 'adx_pos', 'adx_neg']]
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
        module_name, class_name, value_names, method_names = self._ta_map[indicator]
        default_args = (getattr(self.kline, value_name) for value_name in value_names)
        ta_class = getattr(getattr(ta, module_name), class_name)(*default_args, *args, **kwargs)
        computed_indicators = []
        for method_name in method_names:
            computed_indicators.append(getattr(ta_class, method_name)())
        return pd.concat(computed_indicators, axis=1)

    def append(self, computed_indicator, *args, **kwargs):
        """Append an indicator.
        """
        for column in computed_indicator:
            self.loc[:, column.lower()] = computed_indicator[column]
        return None

    def extend(self, indicator, *args, **kwargs):
        """Compute and append an indicator.
        """
        computed_indicator = self.compute(indicator, *args, **kwargs)
        self.append(computed_indicator, *args, **kwargs)
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



class _Metrics(pd.DataFrame):
    """
    Pandas DataFrame subclass with custom metadata and constructor for Metrics class.
    """
    _metadata = ['_kline', '_cached', '_store_metrics']

    @property
    def _constructor(self):
        """Requirement for correct pd.DataFrame subclassing.
        """
        return _Metrics

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
    def store_metrics(self):
        """Get stored metrics list.
        """
        return self._store_metrics

    def compute(self, metric, *args, **kwargs):
        """Compute a metric.
        """
        fun = metric
        computed_metric = getattr(self, '_compute_metric_'+fun)(self.kline, *args, **kwargs)
        return computed_metric

    def append(self, computed_metric, metric, *args, **kwargs):
        """Append a metric.
        """
        str_args = '_' + '_'.join([str(_) for _ in args]) if len(args) > 0 else ''
        str_kwargs = '_' + '_'.join([str(kwargs[_]) for _ in kwargs]) if len(kwargs) > 0 else ''
        metric_name = metric + str_args + str_kwargs
        self.loc[:, metric_name] = computed_metric
        return None

    def extend(self, metric, *args, **kwargs):
        """Compute and append a metric.
        """
        computed_metric = self.compute(metric, *args, **kwargs)
        self.append(computed_metric, metric, *args, **kwargs)
        return None

    @staticmethod
    def _compute_metric_SMA(kline, window=50):
        sma = kline.close.rolling(window).mean()
        return sma

    @staticmethod
    def _compute_metric_EMA(kline, window=50):
        # d = dict()
        multiplier = 2/(window+1)
        ema_yesterday = kline.close.iloc[:window].mean()
        emas = []
        for i in range(len(kline.close)):
            ema_today = (kline.close.iloc[i]*multiplier)+(ema_yesterday*(1-multiplier))
            emas.append(ema_today)
            ema_yesterday = ema_today
        emas = pd.Series(emas, index=kline.index)
        return emas

    @staticmethod
    def _compute_metric_WMA(kline, window=50, data=None):
        data = [] if data is None else data
        p = kline.close if len(data)==0 else pd.Series(data)
        p_wma = pd.Series(data=[0]*len(p), index=p.index)
        # TBO.
        for i,j in zip(range(window), range(window)[::-1]):
            p_wma += p.shift(i)*(j+1)
        return p_wma/(np.sum(range(1,window+1)))

    @staticmethod
    def _compute_metric_HMA(kline, window=50):
        wma_n = getattr(Metrics, '_compute_metric_WMA')(kline=kline, window=window)
        wma_n2 = getattr(Metrics, '_compute_metric_WMA')(kline=kline, window=int(window/2))
        hma = getattr(Metrics, '_compute_metric_WMA')(kline=[], data=(2*wma_n2-wma_n), window=int(np.sqrt(window)))
        return hma

class Metrics(_Metrics):
    """
    Object for asset-specific metrics manipulations.

    Args:
        kline (Kline): Kline from which to get info to compute metrics.
        store_metrics (list): Metric names to store in memory.
    """
    def __init__(self, kline, store_metrics):
        self._kline=kline
        self._cached=False
        self._store_metrics=store_metrics
        # super().__init__()

    def __getattr__(self, attr_name):
        if not self._cached:
            self._cached=True
            data=pd.DataFrame(data=[], index=self.kline.index)
            super(Metrics, self).__init__(data=data)
            for metric in self.store_metrics:
                try:
                    fun, args = metric
                except ValueError:
                    fun = metric
                self.extend(fun, *args)
        return super(Metrics, self).__getattr__(attr_name)




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
    def _compute_signal_CROSSOVERS(kline, fast_metric='SMA_50', slow_metric='SMA_200', buffer=.0001):
        sl_diff = (getattr(kline.metrics, fast_metric) -
                   getattr(kline.metrics, slow_metric))/getattr(kline.metrics, fast_metric)
        # crossovers = ((sl_diff > 0) & (sl_diff > sl_diff.shift(1))).map({True: 1., False: -1.})
        bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        cuts = pd.cut(sl_diff, bins=bins, duplicates='drop', labels=False)-1.
        return cuts.where(cuts != 1, cuts.where(sl_diff.diff(1) > 0, 0))

    @staticmethod
    def _compute_signal_TREND(kline, metric='SMA_50', repeat=2, buffer=.0001):
        ydiff = getattr(kline.metrics, metric).pct_change(1)
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
    def _compute_signal_PRICECROSS(kline, metric='SMA_50', buffer=.0001):
        price_metric_diff = (kline.close - getattr(kline.metrics, metric))/kline.close
        bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        return pd.cut(price_metric_diff, bins=bins, duplicates='drop', labels=False)-1.

    @staticmethod
    def _compute_signal_MACDCROSS(kline, fast_metric='EMA_12', slow_metric='EMA_26', signal='EMA', window=9,
                                  buffer=.0001):
        macd = (getattr(kline.metrics, fast_metric)-getattr(kline.metrics, slow_metric))
        signal_line = getattr(Metrics, '_compute_metric_'+signal)(macd.rename('close').to_frame(), window=window)
        macd_signal_diff = (macd - signal_line)/macd
        return macd_signal_diff
        # bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        # return pd.cut(macd_signal_diff, bins=bins, duplicates='drop', labels=False)-1.

    @staticmethod
    def _compute_signal_MACDCROSS2(kline, fast=12, slow=26, window=9, buffer=.0001):
        macd = getattr(kline.indicators, 'macd_{}_{}'.format(fast, slow))
        macd_diff = getattr(kline.indicators, 'macd_diff_{}_{}'.format(fast, slow))
        macd_signal_diff = macd_diff/macd
        return macd_signal_diff
        # bins = [-float('inf'), -buffer, buffer+1e-10, float('inf')]
        # return pd.cut(macd_signal_diff, bins=bins, duplicates='drop', labels=False)-1.

    @staticmethod
    def _compute_signal_RSICROSS(kline, window=14, lower=30, upper=70):
        pct_gain = kline.close.pct_change().clip(lower=0)
        pct_loss = kline.close.pct_change().clip(upper=0).abs()
        avg_gain = pct_gain.rolling(window).mean()
        avg_loss = pct_loss.rolling(window).mean()
        smoothed_avg_gain = ((avg_gain.shift(1)*(window-1))+pct_gain)/window
        smoothed_avg_loss = ((avg_loss.shift(1)*(window-1))+pct_loss)/window
        rsi = 100-(100/(1+(smoothed_avg_gain/smoothed_avg_loss)))
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
