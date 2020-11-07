'''
Objects for unique assets.
'''
# other
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from requests.exceptions import ReadTimeout
from datetime import datetime

class Kline(pd.DataFrame):
  '''
  Object for asset-specific data storage, retrieval, and analysis.

  Args:
    asset (str): String of the official base-quote acronym.
    url_scheme (type): Function for the desired url-scheme. Defaults to str.
    root_path (str): Root path of the stored data.
    store_metrics (iterable): List of metrics to store.
    store_signals (iterable): List of signals to store.
  '''
  _metadata = ['_asset', '_metrics', '_signals', '_url_scheme', '_root_path', '_store_metrics', '_store_signals',
               '_initialized']
  _cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'n/a']

  def __init__(self, asset, url_scheme=str, root_path='data-bucket/data/', store_metrics=[], store_signals=[]):
    self._asset = asset
    self._url_scheme = url_scheme
    self._root_path = root_path
    self._store_metrics = store_metrics
    self._store_signals = store_signals
    self._initialized = False
    self._metrics = Metrics(self, store_metrics=self._store_metrics)
    self._signals = Signals(self, store_signals=self._store_signals)

  def __getattr__(self, attr_name):
    if self._initialized==False:
      self._initialized=True
      data=self._get_stored()
      kwargs=dict()
      kwargs['columns'] = self._cols
      super(Kline, self).__init__(data=data, dtype=np.float64, **kwargs)
    return super(Kline, self).__getattr__(attr_name)

  @property
  def asset(self):
    return self._asset

  @property
  def name(self):
    return self._asset
  
  @property
  def metrics(self):
    return self._metrics

  @property
  def signals(self):
    return self._signals

  @property
  def url_scheme(self):
    return self._url_scheme

  @property
  def root_path(self):
    return self._root_path

  @property
  def store_metrics(self):
    return self._store_metrics

  @property
  def store_signals(self):
    return self._store_signals

  @property
  def initialized(self):
    return self._initialized

  @property
  def start(self):
    return self.index.min()

  @property
  def end(self):
    return self.index.max()

  def _open(self, path, mode):
    if self.url_scheme==str:
      return self.url_scheme(path)
    else:
      return self.url_scheme(path, mode)

  def _get_stored(self):
    '''
    Returns:
      (pandas.DataFrame) The asset's stored data.
    '''
    try:
      filepath_or_buffer = self._open(self.root_path+self.asset+'.csv', mode='r')
      stored_df = pd.read_csv(filepath_or_buffer=filepath_or_buffer, index_col=0, sep=';', parse_dates=True,
        dtype=np.float64)
    except FileNotFoundError:
      stored_df = None
    return stored_df

  def update(self, client, store=False, verbose=False, sleep=1):
    '''
    Args:
      client: Connection client to the crypto exchange.
      store (bool): If True, overwrites the stored data. Else, overwrites only the data in memory.
      verbose (bool): Controls verbosity.
      sleep (int): Sleep interval between update ticks.

    Returns:
      None. Fetches up-to-date data on the crypto exchange, updates self, and stores it if indicated.
    '''
    today = datetime.utcnow()
    today_ts = pd.Timestamp(year=today.year, month=today.month, day=today.day, hour=today.hour)
    while self.index.max() != today_ts:
      start_from_index = self.index.max()
      start_from_index = pd.Timestamp(year=2017, month=1, day=1, hour=0, tz='UTC') if pd.isnull(start_from_index) else start_from_index
      remote_df = self.get_remote(client=client, start_datetime=start_from_index)
      for index in remote_df.index:
        if verbose:
          print(self.name, index)
        self.loc[index,:] = remote_df.loc[index,:]
      time.sleep(sleep)
    if store:
      self.store()
    self._metrics = Metrics(self, store_metrics=self.store_metrics)
    self._signals = Signals(self, store_signals=self.store_signals)

  def get_remote(self, client, start_datetime): # end_datetime=None
    '''
    Args:
      client: Connection client to the crypto exchange.
      start_datetime (datetime): Starting datetime from when to fetch data.
      end_datetime (datetime): Ending datetime to fetch data up to.

    Returns:
      (pandas.DataFrame): Dataframe of the asset's remote data.
    '''
    start_timestamp = int(start_datetime.timestamp()*1000)
    # if end_datetime is None:
    #   end_datetime = pd.Timestamp.utcnow()
    # end_timestamp = int(end_datetime.timestamp()*1000)
    klines = client.get_historical_klines(symbol=self.asset, interval=client.KLINE_INTERVAL_1HOUR,
                                          start_str=start_timestamp, # end_str=end_timestamp, 
                                          limit=999999999)
    timestamps = [kline[0]/1000 for kline in klines]
    datetimes = [datetime.utcfromtimestamp(timestamp) for timestamp in timestamps]
    remote_df = pd.DataFrame(data=klines, columns=self._cols, index=datetimes, dtype=np.float64)
    return remote_df

  def store(self):
    '''
    Returns:
      None. Overwrites the asset's stored data.
    '''
    path_or_buf = self._open(self.root_path+self.asset+'.csv', mode='w')
    self.to_csv(path_or_buf=path_or_buf, sep=';')

  def plot(self, days=180, signals=None, metrics=None, rangeslider=False, fig=None):
    '''
    Args:
      days (int): The number of days in the past for which to plot the desired graph.
      signals (list): List of signals to plot.
      metrics (list): List of metrics to plot.
      rangeslides (bool): Whether to plot a rangeslider or not.
      fig (plotly.graph_objs.Figure): An exising plotly figure on which to plot.

    Returns:
      None. Plots the assets "open-high-low-close" data in candlestick style,
      overlayed with the desired metrics/signals.   
    '''
    signals = [] if signals is None else signals
    metrics = [] if metrics is None else metrics
    y_min, y_max = np.percentile(self.open[-days*24:], [0,100])
    # colors = ['rgb(0, 11, 24)','rgb(0, 23, 45)','rgb(0, 38, 77)','rgb(2, 56, 110)','rgb(0, 73, 141)','rgb(0, 82, 162)']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = ["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590", "#277da1"]
    # ohlc = self.loc[:,['open','high','low','close']]
    fig = go.Figure() if fig==None else fig
    fig.add_trace(go.Candlestick(x=self.index[-days*24:], open=self.open[-days*24:], high=self.high[-days*24:],
                                 low=self.low[-days*24:], close=self.close[-days*24:], name=self.name))
    for i, metric in enumerate(metrics):
      color_index = int(i/len(metrics)*len(colors))
      fig.add_trace(go.Scatter(x=self.index[-days*24:], y=self.metrics.loc[:, metric][-days*24:], name=metric,
                               line=dict(color=colors[color_index])))
    for signal in signals:
      fig.add_trace(go.Scatter(x=self.index[-days*24:], y=np.where(self.signals.loc[:,signal][-days*24:],y_max,0),
                               mode='none', fill='tozeroy', fillcolor='rgba(60,220,100,0.2)', name=signal))
    fig.update_layout(xaxis_rangeslider_visible=rangeslider)
    fig.update_yaxes(range=[y_min, y_max])
    fig.show()


class Metrics(pd.DataFrame):
  _metadata = ['_kline', '_initialized', '_store_metrics']
  def __init__(self, kline, store_metrics):
    self._kline=kline
    self._initialized=False
    self._store_metrics=store_metrics

  @property
  def kline(self):
    return self._kline

  @property
  def initialized(self):
    return self._initialized

  @property
  def store_metrics(self):
    return self._store_metrics

  def __getattr__(self, attr_name):
    if self._initialized==False:
      self._initialized=True
      data=pd.DataFrame(data=[], index=self.kline.index, columns=self.store_metrics)
      super(Metrics, self).__init__(data=data)
      for metric in self.store_metrics:
        self.compute(metric=metric, append=True)
    return super(Metrics, self).__getattr__(attr_name)

  def compute(self, metric, append=False, *args, **kwargs):
    try:
      fun, arg = metric.split('_')
      args = (int(arg),)
    except ValueError:
      fun = metric
    computed_metric = getattr(self, '_compute_metric_'+fun)(self.kline, *args, **kwargs)
    if append==True:
      metric_name = metric
      self.loc[:, metric_name] = computed_metric
      return None
    return computed_metric

  @staticmethod
  def _compute_metric_SMA(kline, window):
    sma = kline.close.rolling(window).mean()
    return sma

  @staticmethod
  def _compute_metric_EMA(kline, window):
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
  def _compute_metric_WMA(kline, window, data=[]):
    p = kline.close  if len(data)==0 else pd.Series(data)
    p_wma = pd.Series(data=[0]*len(p), index=p.index)
    # TBO.
    for i,j in zip(range(window), range(window)[::-1]):
      p_wma += p.shift(i)*(j+1)
    return p_wma/(np.sum(range(1,window+1)))

  @staticmethod
  def _compute_metric_HMA(kline, window):
    wma_n = Metrics._compute_metric_WMA(kline=kline, window=window)
    wma_n2 = Metrics._compute_metric_WMA(kline=kline, window=int(window/2))
    hma = Metrics._compute_metric_WMA(kline=[], data=(2*wma_n2-wma_n), window=int(np.sqrt(window)))
    return hma


class Signals(pd.DataFrame):
  _metadata = ['_kline', '_initialized', '_store_signals']

  def __init__(self, kline, store_signals):
    self._kline=kline
    self._initialized=False
    self._store_signals = store_signals

  @property
  def kline(self):
    return self._kline

  @property
  def initialized(self):
    return self._initialized

  @property
  def store_signals(self):
    return self._store_signals

  def __getattr__(self, attr_name):
    if self._initialized==False:
      self._initialized=True
      data=pd.DataFrame(data=[], index=self.kline.index, columns=self.store_signals)
      super(Signals, self).__init__(data=data)
      for signal in self.store_signals:
        self.compute(signal=signal, append=True)
    return super(Signals, self).__getattr__(attr_name)

  def compute(self, signal, append=False, *args, **kwargs):
    fun = signal
    computed_signal = getattr(self, '_compute_signal_'+fun)(self.kline, *args, **kwargs)
    if append==True:
      signal_name = signal
      self.loc[:, signal_name] = computed_signal
      return None
    return computed_signal

  @staticmethod
  def _compute_signal_SMA_CROSSOVERS(kline):
    sl_diff = kline.metrics.SMA_50 - kline.metrics.SMA_200
    crossovers = (sl_diff>0) & (sl_diff>sl_diff.shift(1))
    return crossovers

  @staticmethod
  def _compute_signal_SMA_TREND(kline, consecutive_days=2, min_slope=1e-7):
    ydiff = kline.metrics.SMA_50 - kline.metrics.SMA_50.shift(1)
    dateindex = pd.Series(kline.index.values)
    xdiff = (dateindex - dateindex.shift(1)).apply(lambda x :x.days)
    pos_slope = pd.Series(((ydiff.values/xdiff.values)>min_slope).astype(int), index=kline.index)
    pos_slope_cumul = pos_slope * (pos_slope.groupby((pos_slope != pos_slope.shift()).cumsum()).cumcount() + 1)
    trend = pos_slope_cumul>(consecutive_days-1)
    return trend

  @staticmethod
  def _compute_signal_SMA_PRICECROSS(kline):
    pricecross = kline.close > kline.metrics.SMA_50
    return pricecross
