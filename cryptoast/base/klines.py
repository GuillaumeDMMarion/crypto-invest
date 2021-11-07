"""
Objects for collections of assets.
"""
# from datetime import datetime
from typing import List, Optional, Callable
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from requests.exceptions import ReadTimeout

from cryptoast.base.kline import Kline


class Klines(dict):
    """
    Object for regrouping multiple Kline objects, and representing them as a readable list of official asset acronyms.

    Args:
        klines: A collection of multiple Kline objects.
    """

    def __init__(self, klines: Optional[List] = None):
        klines = {} if klines is None else klines
        if isinstance(klines, (dict, type(Klines))):
            super(Klines, self).__init__(klines)
        else:
            super(Klines, self).__init__(zip([kline.asset for kline in klines], klines))
        try:
            for col in (self.listedvalues()[0].columns.tolist()) + [
                "indicators",
                "signals",
            ]:
                self._add_dynamic_fct(col)
        except IndexError:
            pass

    def __repr__(self) -> str:
        repr_l = [str(k) + ": <class 'Kline'>" for k, v in self.sorteditems()]
        repr_s = "{" + ", ".join(repr_l) + "}"
        return str(repr_s)

    def __str__(self) -> str:
        str_l = [str(k) + ": " + str(v.close.iloc[-1]) for k, v in self.sorteditems()]
        str_s = "\n".join(str_l)
        return str(str_s)

    def _add_dynamic_fct(self, col: str) -> None:
        """Add dynamic function."""

        def dynamic_fct(index=-1, func=np.mean):
            group = lambda x: func(x)
            return dict(
                [
                    (asset, group(getattr(kline, col).iloc[index]))
                    for asset, kline in self.sorteditems()
                ]
            )

        dynamic_fct.__doc__ = """Args:\n    index (int/slice): Index or slice of the line(s) to include.
            func (function): Function to apply.\n\nReturns:
            func aggregate of {} over the indicated index for all
          assets.""".format(
            col
        )
        dynamic_fct.__name__ = "{}".format(col)
        setattr(self, dynamic_fct.__name__, dynamic_fct)

    def sorteditems(self) -> List:
        """
        Returns:
            A collection of the key-sorted key, value tuples.
        """
        return sorted(self.items(), key=lambda x: x[0])

    def listedvalues(self) -> List:
        """
        Returns:
            A collection of the values.
        """
        return list(self.values())

    def sortedkeys(self) -> List:
        """
        Returns:
            A collection of the sorted keys.
        """
        return sorted(self.keys())

    def reindex(self, index: np.ndarray, fill: str = "ffill") -> "Klines":
        """
        Returns:
            A reindexed (collection of) Kline oject(s).
        """
        reindexes = [v.reindex(index) for k, v in self.items()]
        if fill is not None:
            reindexes = [getattr(reindex, fill)() for reindex in reindexes]
        return Klines(reindexes)

    def select(self, names: List, method: str = "quote") -> "Kline":
        """
        Args:
            names: The asset or quote acronyms.
            method: Whether to select on 'base', 'quote' or 'asset'.

        Returns:
            A new (collection of) Kline object(s).
        """
        names = (names,) if isinstance(names, str) else names
        fun_dict = {
            "base": str.startswith,
            "quote": str.endswith,
            "asset": np.random.choice([str.startswith, str.endswith]),
        }
        fun = fun_dict[method]
        selection = [[v for k, v in self.items() if fun(k, name)] for name in names]
        flat_selection = [item for sublist in selection for item in sublist]
        return Klines(flat_selection)


class KLMngr(Klines):
    """
    Management object for all available assets.

    Args:
        quotes_or_assets: Acronyms of either quotes or the full base-quote symbols. Supersedes klines param.
        klines: A selection of klines.
        client: Connection client to the crypto exchange.
        url_scheme: Function for the desired url-scheme. Defaults to str.
        root_path: Root path of the stored data.
        kwargs: Any other kwargs to pass to Kline instantiation.
    """

    _metadata_path = "metadata/info.csv"
    _info_cols = [
        "symbol",
        "base",
        "quote",
        "status",
        "minPrice",
        "maxPrice",
        "tickSize",
        "minQty",
        "maxQty",
        "stepSize",
        "last_update",
    ]

    def __init__(
        self,
        quotes_or_assets: Optional[List] = None,
        klines: Optional[List] = None,
        client: Optional["binance.client.Client"] = None,
        url_scheme: Callable = str,
        root_path: str = "data/",
        **kwargs
    ):
        self._quotes_or_assets = quotes_or_assets
        self._client = client
        self._url_scheme = url_scheme
        self._root_path = root_path
        self._kline_kwargs = kwargs
        self._get_info()
        self._bmpc = None
        if quotes_or_assets is not None:
            self.from_quotes_or_assets(quotes_or_assets)
        elif klines is not None:
            self.from_klines(klines)

    @property
    def quotes_or_assets(self) -> List:
        """Get quotes or assets list."""
        return self._quotes_or_assets

    @property
    def client(self) -> "binance.client.Client":
        """Get client."""
        return self._client

    @property
    def url_scheme(self) -> Callable:
        """Get url scheme."""
        return self._url_scheme

    @property
    def root_path(self) -> str:
        """Get rooth path."""
        return self._root_path

    @property
    def info(self) -> pd.Series:
        """Get klines info."""
        return self._info

    @property
    def kline_kwargs(self) -> dict:
        """Get klines kline_kwargs."""
        return self._kline_kwargs

    @property
    def bmpc(self) -> pd.DataFrame:
        """Get klines bull-market-percentage components."""
        return self._bmpc

    @property
    def bmp(self) -> pd.Series:
        """Get klines bmp."""
        return self._bmpc.mean(axis=1)

    @property
    def assets(self) -> List:
        """Get assets list."""
        return self.sortedkeys()

    def from_klines(self, klines: List) -> "KLMNgr":
        """
        Args:
            klines: List of klines to initialize.

        Returns:
            The initialized KLMngr.
        """
        return super(KLMngr, self).__init__(klines)

    def from_quotes_or_assets(self, quotes_or_assets: List) -> "KLMNgr":
        """
        Args:
            quotes_or_assets: Acronyms of either quotes or the full base-quote symbols.

        Returns:
            The initialized KLMngr.
        """
        assets = self.find_assets(quotes_or_assets)
        klines = [
            Kline(
                asset=asset,
                url_scheme=self.url_scheme,
                root_path=self.root_path + "data/",
                info=self.info.loc[asset],
                **self.kline_kwargs
            )
            for asset in assets
        ]
        return self.from_klines(klines)

    def find_assets(self, quotes_or_assets: List) -> List:
        """
        Args:
            quotes_or_assets: Acronyms of either quotes or the full base-quote symbols.

        Returns:
            All assets resulting from the intersection between the provided argument and the list of all existing
            assets from the meta-data on the remote server.
        """
        assets_match = [
            asset for asset in self.info.index.values if asset in quotes_or_assets
        ]
        if len(assets_match) > 0:
            return assets_match
        return [
            asset
            for asset in self.info.index.values
            if self.info.loc[asset, "quote"] in quotes_or_assets
        ]

    def compute_bmp(self, components: bool = False) -> pd.DataFrame:
        """
        Args:
            components: Whether or not to return the indiviual components.

        Returns:
            Bull-market percentage for all the initialized assets.
        """
        date_range = pd.date_range(
            "2017-01-01 00:00:00", self.info.last_update.dropna().max(), freq="H"
        )
        df_bmpc = pd.DataFrame(data=[], index=date_range)
        assets = self.sortedkeys()
        for asset in assets:
            avg_asset = self[asset].signals.mean(axis=1)
            df_bmpc.loc[avg_asset.index.round("H"), asset] = avg_asset
        bmpc = df_bmpc.dropna(axis=0, how="all")
        self._bmpc = bmpc
        if not components:
            return self.bmpc
        return self.bmp

    def plot_bmp(self, fig: Optional[Figure] = None) -> Figure:
        """
        Args:
            fig: Plotly Figure.

        Returns:
            Graph of the bull-market percentage for all the initialized assets.
        """
        if self.bmpc is None:
            self.compute_bmp()
        bmp = self.bmp
        fig = go.Figure() if fig is None else fig
        fig.add_trace(
            go.Scatter(x=bmp.index, y=bmp.values, name="Bull Market Percentage")
        )
        fig.show()

    def update(self, verbose: int = 1, sleep: int = 1) -> None:
        """Updates for all initialized assets the data and for all assets existing on the remote server the meta-data.

        Args:
            verbose: Controls verbosity.
            sleep: Sleep interval between update ticks.
        """
        self.update_info()
        self.update_data(verbose=verbose, sleep=sleep)

    @staticmethod
    def _parse_symbols_info(info: dict) -> List:
        """Parses exchange info to readable format."""
        parse_fltr_line = lambda l, f: [
            _ for _ in l["filters"] if _["filterType"] == f
        ][0]
        parse_info_line = lambda l: (
            l["symbol"],
            l["baseAsset"],
            l["quoteAsset"],
            l["status"],
            float(parse_fltr_line(l, "PRICE_FILTER")["minPrice"]),
            float(parse_fltr_line(l, "PRICE_FILTER")["maxPrice"]),
            float(parse_fltr_line(l, "PRICE_FILTER")["tickSize"]),
            float(parse_fltr_line(l, "LOT_SIZE")["minQty"]),
            float(parse_fltr_line(l, "LOT_SIZE")["maxQty"]),
            float(parse_fltr_line(l, "LOT_SIZE")["stepSize"]),
        )
        return [parse_info_line(l) for l in info]

    def update_info(self) -> None:
        """Gets the updated metadata for all assets existing on the crypto exchange, and store it."""
        exchange_info = self.client.get_exchange_info()
        symbols_info = exchange_info["symbols"]
        symbols_info_parsed = KLMngr._parse_symbols_info(symbols_info)
        new_info = pd.DataFrame(
            symbols_info_parsed, columns=self._info_cols[:-1]
        ).set_index("symbol")
        new_info.loc[:, "last_update"] = pd.Series(dtype="<M8[ns]")
        both_new_and_old = set(new_info.index) & set(self.info.index)
        new_info.loc[both_new_and_old, "last_update"] = self.info.loc[
            both_new_and_old, "last_update"
        ]
        setattr(self, "_info", new_info)
        path_or_buf = self._open(self.root_path + self._metadata_path, mode="w")
        self.info.to_csv(path_or_buf=path_or_buf, sep=";")

    def update_data(self, verbose: int, sleep: int, retries: int = 5) -> None:
        """Updates for all initialized assets the data and writes it to csv.

        Args:
            verbose: Controls verbosity.
            sleep: Sleep interval between update ticks.
            retries: Number of retry if TimeoutError is encountered.
        """
        self.from_quotes_or_assets(self.sortedkeys())
        progress_func = tqdm if verbose == 1 else list
        for asset in progress_func(self.sortedkeys()):
            retry = 0
            while True:
                try:
                    verbose_asset = verbose == 2
                    self[asset].update(
                        self.client, store=True, verbose=verbose_asset, sleep=sleep
                    )
                    break
                except (TimeoutError, ReadTimeout):
                    retry += 1
                if retry == retries:
                    break
            self._info.loc[asset, "last_update"] = self[asset].index[-1]
            path_or_buf = self._open(self.root_path + self._metadata_path, mode="w")
            self.info.to_csv(path_or_buf=path_or_buf, sep=";")

    def _get_info(self) -> None:
        """Reads info if it is present. Updates it if not."""
        try:
            filepath_or_buffer = self._open(
                self.root_path + self._metadata_path, mode="r"
            )
            self._info = pd.read_csv(
                filepath_or_buffer=filepath_or_buffer,
                sep=";",
                index_col=0,
                parse_dates=["last_update"],
            )
        except FileNotFoundError:
            self._info = pd.DataFrame([], columns=self._info_cols)
            self.update_info()

    def _open(self, path, mode):
        if self.url_scheme == str:
            return self.url_scheme(path)
        else:
            return self.url_scheme(path, mode)
