"""
Objects for collections of assets.
"""
# from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from requests.exceptions import ReadTimeout

from cryptoast.base.kline import Kline




class Klines(dict):
    """
    Object for regrouping multiple Kline objects, and representing them as a readable list of official asset acronyms.

    Args:
        klines (list): A collection of multiple Kline objects.
    """
    def __init__(self, klines=None):
        klines = {} if klines is None else klines
        if isinstance(klines, (dict, type(Klines))):
            super(Klines, self).__init__(klines)
        else:
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
        """
        Returns:
            (list) A collection of the key-sorted key, value tuples.
        """
        return sorted(self.items(), key=lambda x: x[0])

    def listedvalues(self):
        """
        Returns:
            (list) A collection of the values.
        """
        return list(self.values())

    def sortedkeys(self):
        """
        Returns:
            (list) A collection of the sorted keys.
        """
        return sorted(self.keys())

    def select(self, names, method='quote'):
        """
        Args:
            names (list): The asset or quote acronyms.
            method (str): Whether to select on 'base', 'quote' or 'asset'.

        Returns:
            (Klines) A new (collection of) Kline object(s).
        """
        names = (names,) if isinstance(names, str) else names
        fun_dict = {'base':str.startswith, 'quote':str.endswith, 'asset':np.random.choice([str.startswith,
                                                                                           str.endswith])}
        fun = fun_dict[method]
        selection=[[v for k,v in self.items() if fun(k, name)] for name in names]
        flat_selection=[item for sublist in selection for item in sublist]
        return Klines(flat_selection)


class KLMngr(Klines):
    """
    Management object for all available assets.

    Args:
        quotes_or_assets (list): Acronyms of either quotes or the full base-quote symbols. Supersedes klines param.
        klines (list): A selection of klines.
        client: Connection client to the crypto exchange.
        url_scheme (type): Function for the desired url-scheme. Defaults to str.
        root_path (str): Root path of the stored data.
        store_metrics (iterable): List of metrics to store.
        store_signals (iterable): List of signals to store.
    """
    _metadata_path = 'metadata/info.csv'
    _info_cols = ['symbol', 'base', 'quote', 'status', 'minPrice', 'maxPrice', 'tickSize', 'minQty', 'maxQty',
                  'stepSize']

    def __init__(self, quotes_or_assets=None, klines=None, client=None, url_scheme=str, root_path='data-bucket/',
                             store_metrics=None, store_signals=None):
        self._quotes_or_assets = quotes_or_assets
        self._client = client
        self._url_scheme = url_scheme
        self._root_path = root_path
        self._store_metrics = store_metrics
        self._store_signals = store_signals
        self._get_info()
        if quotes_or_assets is not None:
            self.from_quotes_or_assets(quotes_or_assets)
        elif klines is not None:
            self.from_klines(klines)
        try:
            for col in (self.listedvalues()[0].columns.tolist())+['metrics', 'signals']:
                self._add_dynamic_fct(col)
        except IndexError:
            pass

    @property
    def quotes_or_assets(self):
        """Get quotes or assets list.
        """
        return self._quotes_or_assets

    @property
    def client(self):
        """Get client.
        """
        return self._client

    @property
    def url_scheme(self):
        """Get url scheme.
        """
        return self._url_scheme

    @property
    def root_path(self):
        """Get rooth path.
        """
        return self._root_path

    @property
    def info(self):
        """Get klines info.
        """
        return self._info

    @property
    def assets(self):
        """Get assets list.
        """
        return self.sortedkeys()

    def from_klines(self, klines):
        """
        Args:
            klines (list): List of klines to initialize.

        Returns:
            The initialized KLMngr.
        """
        return super(KLMngr, self).__init__(klines)

    def from_quotes_or_assets(self, quotes_or_assets):
        """
        Args:
            quotes_or_assets (list): Acronyms of either quotes or the full base-quote symbols.

        Returns:
            The initialized KLMngr.
        """
        assets = self.find_assets(quotes_or_assets)
        klines = [Kline(asset=asset, url_scheme=self._url_scheme, root_path=self._root_path+'data/',
                                        store_metrics=self._store_metrics,
                                        store_signals=self._store_signals,
                                        info=self.info.loc[asset])
                            for asset in assets]
        return self.from_klines(klines)

    def find_assets(self, quotes_or_assets):
        """
        Args:
            quotes_or_assets (list): Acronyms of either quotes or the full base-quote symbols.

        Returns:
            (list) All assets resulting from the intersection between the provided argument and the list of all existing
                         assets from the meta-data on the remote server.
        """
        assets_match = [asset for asset in self.info.index.values if asset in quotes_or_assets]
        if len(assets_match) > 0:
            return assets_match
        return [asset for asset in self.info.index.values if self.info.loc[asset, 'quote'] in quotes_or_assets]

    def get_bmp(self, components=False):
        """
        Returns:
            (pd.DataFrame) Bull-market percentage for all the initialized assets.
        """
        date_range = pd.date_range('2017-01-01 00:00:00', self.info.last_update.dropna().max(), freq='H')
        df_bmp = pd.DataFrame(data=[], index=date_range)
        assets = self.sortedkeys()
        for asset in assets:
            avg_asset = self[asset].signals.mean(axis=1)
            df_bmp.loc[avg_asset.index.round('H'), asset] = avg_asset
        bmp = df_bmp.dropna(axis=0, how='all')
        if not components:
            bmp = bmp.mean(axis=1)
        return bmp

    def plot_bmp(self, bmp=None, fig=None):
        """
        Returns:
            (plotly.graph_objects.Figure) Graph of the bull-market percentage for all the initialized assets.
        """
        bmp = self.get_bmp() if bmp is None else bmp
        fig = go.Figure() if fig is None else fig
        fig.add_trace(go.Scatter(x=bmp.index, y=bmp.values, name='Bull Market Percentage'))
        fig.show()

    def update(self, verbose=1, sleep=1):
        """
        Args:
            verbose (int): Controls verbosity.
            sleep (int): Sleep interval between update ticks.

        Returns:
            None. Updates for all initialized assets the data and for all assets existing on the remote server the
                  meta-data.
        """
        self.update_info()
        self.update_data(verbose=verbose, sleep=sleep)

    @staticmethod
    def _parse_symbols_info(info):
        parse_fltr_line = lambda l, f: [_ for _ in l['filters'] if _['filterType'] == f][0]
        parse_info_line = lambda l: (l['symbol'], l['baseAsset'], l['quoteAsset'], l['status'],
                                     float(parse_fltr_line(l, 'PRICE_FILTER')['minPrice']),
                                     float(parse_fltr_line(l, 'PRICE_FILTER')['maxPrice']),
                                     float(parse_fltr_line(l, 'PRICE_FILTER')['tickSize']),
                                     float(parse_fltr_line(l, 'LOT_SIZE')['minQty']),
                                     float(parse_fltr_line(l, 'LOT_SIZE')['maxQty']),
                                     float(parse_fltr_line(l, 'LOT_SIZE')['stepSize']))
        return [parse_info_line(l) for l in info]

    def update_info(self):
        """
        Returns:
            None. Gets the updated metadata for all assets existing on the crypto exchange, and store it.
        """
        exchange_info = self.client.get_exchange_info()
        symbols_info = exchange_info['symbols']
        symbols_info_parsed = KLMngr._parse_symbols_info(symbols_info)
        new_info = pd.DataFrame(symbols_info_parsed, columns=self._info_cols).set_index('symbol')
        new_info.loc[:, 'last_update'] = pd.Series(dtype='<M8[ns]')
        both_new_and_old = set(new_info.index) & set(self.info.index)
        new_info.loc[both_new_and_old, 'last_update'] = self.info.loc[both_new_and_old, 'last_update']
        setattr(self, '_info', new_info)
        path_or_buf = self._open(self._root_path+self._metadata_path, mode='w')
        self._info.to_csv(path_or_buf=path_or_buf, sep=';')

    def update_data(self, verbose, sleep):
        """
        Args:
            verbose (int): Controls verbosity.
            sleep (int): Sleep interval between update ticks.

        Returns:
            None. Updates for all initialized assets the data and writes it to csv.
        """
        self.from_quotes_or_assets(self.sortedkeys())
        progress_func = tqdm if verbose==1 else list
        for asset in progress_func(self.sortedkeys()):
            while True:
                try:
                    verbose_asset = verbose==2
                    self[asset].update(self.client, store=True, verbose=verbose_asset, sleep=sleep)
                    break
                except (TimeoutError, ReadTimeout):
                    pass
            self._info.loc[asset,'last_update'] = self[asset].index[-1]
            path_or_buf = self._open(self._root_path+self._metadata_path, mode='w')
            self._info.to_csv(path_or_buf=path_or_buf, sep=';')

    def _get_info(self):
        """
        Returns:
            None. Reads info if it is present. Updates it if not.
        """
        try:
            filepath_or_buffer = self._open(self._root_path+self._metadata_path, mode='r')
            self._info = pd.read_csv(filepath_or_buffer=filepath_or_buffer, sep=';', index_col=0,
                                     parse_dates=['last_update'])
        except FileNotFoundError:
            self._info = pd.DataFrame([], columns=self._info_cols)
            self.update_info()

    def _add_dynamic_fct(self, col):
        def dynamic_fct(index=-1, func=np.mean):
            group = lambda x: func(x)
            return dict([(asset, group(getattr(kline, col).iloc[index])) for asset, kline in self.sorteditems()])
        dynamic_fct.__doc__ = """Args:\n    index (int/slice): Index or slice of the line(s) to include.
            func (function): Function to apply.\n\nReturns:
            func aggregate of {} over the indicated index for all
          assets.""".format(col)
        dynamic_fct.__name__ = '{}'.format(col)
        setattr(self, dynamic_fct.__name__, dynamic_fct)

    def _open(self, path, mode):
        if self._url_scheme==str:
            return self._url_scheme(path)
        else:
            return self._url_scheme(path, mode)
