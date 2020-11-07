'''
Objects for collections of assets.
'''
# local
from cryptoast.kline import Kline
# other
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from requests.exceptions import ReadTimeout
from binance.client import Client
from datetime import datetime
from tqdm import tqdm

class Klines(dict):
  '''
  Object for regrouping multiple Kline objects, and representing them as a readable list of official asset acronyms.

  Args:
    klines (list): A collection of multiple Kline objects. 
  '''
  def __init__(self, klines):
    super(Klines, self).__init__(zip([kline.asset for kline in klines], klines))

  def __repr__(self):
    repr_l = [str(k)+": <class 'Kline'>" for k,v in self.sorteditems()]
    repr_s = '{'+', '.join(repr_l)+'}'
    return str(repr_s)

  def __str__(self):
    str_l = [str(k)+': '+str(v.close.iloc[-1]) for k,v in self.sorteditems()]
    str_s = '\n'.join(str_l)
    return str(str_s)

  def sorteditems(self):
    '''
    Returns:
      (list) A collection of the key-sorted key, value tuples.
    '''
    return sorted(self.items(), key=lambda x: x[0])

  def listedvalues(self):
    '''
    Returns:
      (list) A collection of the values.
    '''
    return list(self.values())

  def sortedkeys(self):
    '''
    Returns:
      (list) A collection of the sorted keys.
    '''
    return sorted(self.keys())

  def select(self, names, method='quote'):
    '''
    Args:
      names (list): The asset or quote acronyms.
      method (str): Whether to select on 'base', 'quote' or 'asset'.
    
    Returns:
      (Klines) A new (collection of) Kline object(s).
    '''
    names = (names,) if isinstance(names, str) else names
    fun_dict = {'base':str.startswith, 'quote':str.endswith, 'asset':np.random.choice([str.startswith, str.endswith])}
    fun = fun_dict[method]
    selection=[[v for k,v in self.items() if fun(k, name)] for name in names]
    flat_selection=[item for sublist in selection for item in sublist]
    return Klines(flat_selection)


class KLMngr(Klines):
  '''
  Management object for all available assets.

  Args:
    quotes_or_assets (list): Acronyms of either quotes or the full base-quote symbols. Supersedes klines param.
    klines (list): A selection of klines.
    client: Connection client to the crypto exchange.
    url_scheme (type): Function for the desired url-scheme. Defaults to str.
    root_path (str): Root path of the stored data.
  '''
  def __init__(self, quotes_or_assets=None, klines=None, client=None, url_scheme=str,
               root_path='data-bucket/'):
    self._quotes_or_assets = quotes_or_assets
    self._client = client
    self._url_scheme = url_scheme
    self._root_path = root_path
    self._get_info()
    if quotes_or_assets is not None:
      self.from_quotes_or_assets(quotes_or_assets)
    elif klines is not None:
      self.from_klines(klines)
    try:
      for col in self.listedvalues()[0].columns:
        self._add_dynamic_fct(col)
    except IndexError:
      pass

  @property
  def quotes_or_assets(self):
    return self._quotes_or_assets

  @property
  def client(self):
    return self._client

  @property
  def url_scheme(self):
    return self._url_scheme

  @property
  def root_path(self):
    return self._root_path

  @property
  def info(self):
    return self._info

  @property
  def assets(self):
    return self.sortedkeys()

  def from_klines(self, klines):
    '''
    Args:
      klines (list): List of klines to initialize.

    Returns:
      The initialized KLMngr.
    '''
    return super(KLMngr, self).__init__(klines)

  def from_quotes_or_assets(self, quotes_or_assets):
    '''
    Args:
      quotes_or_assets (list): Acronyms of either quotes or the full base-quote symbols.

    Returns:
      The initialized KLMngr.
    '''
    assets = self.find_assets(quotes_or_assets)
    klines = [Kline(asset=asset, url_scheme=self._url_scheme, root_path=self._root_path+'data/', store_metrics=['SMA_200',
                    'SMA_50','EMA_50','WMA_50','HMA_50'], store_signals=['SMA_CROSSOVERS', 'SMA_TREND', 'SMA_PRICECROSS']) 
              for asset in assets]
    return self.from_klines(klines)

  def find_assets(self, quotes_or_assets):
    '''
    Args:
      quotes_or_assets (list): Acronyms of either quotes or the full base-quote symbols.

    Returns:
      (list) All assets resulting from the intersection between the provided argument and the list of all existing 
             assets from the meta-data on the remote server.
    '''
    assets_match = [asset for asset in self.info.index.values if asset in quotes_or_assets]
    if len(assets_match)>0:
      return assets_match
    return [asset for asset in self.info.index.values if self.info.loc[asset, 'quote'] in quotes_or_assets]

  def get_bmp(self):
    '''
    Returns:
      (pd.DataFrame) Bull-market percentage for all the initialized assets.
    '''
    date_range = pd.date_range('2017-01-01',datetime.today().date(), freq='D')
    df_bmp = pd.DataFrame(data=[], index=date_range)
    assets = self.sortedkeys()
    for asset in assets:
      avg_asset = self[asset].signals.mean(axis=1)#>0
      df_bmp.loc[:,asset] = avg_asset
    df_bmp.dropna(axis=0, how='all', inplace=True)
    bmp = df_bmp.mean(axis=1)
    return bmp

  def plot_bmp(self, fig=None):
    '''
    Returns:
      (plotly.graph_objects.Figure) Graph of the bull-market percentage for all the initialized assets.
    '''
    bmp = self.get_bmp()
    fig = go.Figure() if fig==None else fig
    fig.add_trace(go.Scatter(x=bmp.index, y=bmp.values, name='Bull Market Percentage'))
    fig.show()

  def update(self, verbose=1):
    '''
    Args:
      verbose (int): Controls verbosity.

    Returns:
      None. Updates for all initialized assets the data and for all assets existing on the remote server the meta-data.
    '''
    self._update_info()
    self._update_data(verbose=verbose)

  def _update_info(self):
    '''
    Returns:
      None. Gets the updated metadata for all assets existing on the crypto exchange, and store it.
    '''
    exchange_info = self.client.get_exchange_info()
    exchange_info_list = exchange_info['symbols']
    d = dict()
    keys = ['symbol','quoteAsset','baseAsset','status']
    for key in keys:
      d[key] = [_[key] for _ in exchange_info_list]
    new_info = pd.DataFrame(index=d['symbol'],
                            data={'base':d['baseAsset'], 'quote':d['quoteAsset'],'status':d['status'],
                                  'last_update':[None for _ in range(len(d['symbol']))]})
    both_new_and_old = set(new_info.index) & set(self.info.index)
    new_info.loc[both_new_and_old, 'last_update'] = self.info.loc[both_new_and_old, 'last_update']
    self._info = new_info
    path_or_buf = self._open(self._root_path+'metadata/info.csv', mode='w')
    self._info.to_csv(path_or_buf=path_or_buf, sep=';')

  def _update_data(self, verbose):
    '''
    Args:
      verbose (int): Controls verbosity.

    Returns:
      None. Updates for all initialized assets the data and writes it to csv.
    '''
    self.from_quotes_or_assets(self.sortedkeys())
    progress_func = tqdm if verbose==1 else list
    for asset in progress_func(self.sortedkeys()):
      while True:
        try:
          verbose_asset = verbose==2
          self[asset].update(self.client, store=True, verbose=verbose_asset)
          break
        except (TimeoutError, ReadTimeout):
          pass
      self._info.loc[asset,'last_update'] = self[asset].index[-1]
      path_or_buf = self._open(self._root_path+'metadata/info.csv', mode='w')
      self._info.to_csv(path_or_buf=path_or_buf, sep=';')

  def _get_info(self):
    '''
    Returns:
      None. Reads info if it is present. Updates it if not.
    '''
    try:
      filepath_or_buffer = self._open(self._root_path+'metadata/info.csv', mode='r')
      self._info = pd.read_csv(filepath_or_buffer=filepath_or_buffer, sep=';', index_col=0)[['base','quote','status',
                                                                                             'last_update']]
    except:
      self._info = pd.DataFrame([], columns=['base','quote','status','last_update'])
      self._update_info()

  def _add_dynamic_fct(self, col):
    def dynamic_fct(index=-1, group_fun=np.mean):
      if type(index)==int:
        group = lambda x: x
      else:
        group = lambda x: group_fun(x)
      return dict([(asset, group(kline.loc[:, col].iloc[index])) for asset, kline in self.sorteditems()])
    dynamic_fct.__doc__ = 'Args:\n  index (int/slice): Index or slice of the line(s) to include.\n  group_fun (function): Function to apply.\n\nReturns:\n  group_fun aggregate of {} over the indicated index for all assets.'.format(col)
    dynamic_fct.__name__ = '{}'.format(col)
    setattr(self, dynamic_fct.__name__, dynamic_fct)

  def _open(self, path, mode):
    if self._url_scheme==str:
      return self._url_scheme(path)
    else:
      return self._url_scheme(path, mode)
