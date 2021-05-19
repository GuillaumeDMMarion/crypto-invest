"""
Environments for reinforcement learning.
"""
# other
from datetime import timedelta
import decimal
import gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
# cryptoast
# from cryptoast.base.klines import KLMngr




def get_precision(dec):
    """Get precision.
    """
    d = decimal.Decimal(str(dec))
    return abs(d.as_tuple().exponent)

def prec_ceil(a, precision=0):
    """Precision ceil.
    """
    return np.round(a + 0.5 * 10**(-precision), precision)

def prec_floor(a, precision=0):
    """Precision floor.
    """
    return np.round(a - 0.5 * 10**(-precision), precision)


class Backtest():
    """Backtester.
    """
    def __init__(self, kline=None, start_index=0, init_assets=1, init_cash=0,
                 commission=0.001, slippage_pct=0.01, slippage_steps=0,
                 memory=24):
        self.start_index = start_index
        self.init_assets = init_assets
        self.init_cash = init_cash
        self.commission = min(commission, 1)
        self.slippage_pct = min(slippage_pct, 1)
        self.slippage_steps = slippage_steps
        self.memory = memory
        if kline is not None:
            self.from_kline(kline)

    @property
    def orders(self):
        """Get orders.
        """
        # ['dtm', 'cash', 'assets', 'value']
        return pd.DataFrame(data=self._orders).T

    @property
    def periodic(self):
        """Get periodic.
        """
        # ['dtm', 'dtm_booking', 'filled', 'action', 'price', 'desired_size', 'size', 'commission', 'value']
        return pd.DataFrame(data=self._periodic).T

    @property
    def history(self):
        """Get history.
        """
        return self._history[0]

    @staticmethod
    def _step(order, position, timestamps, periodic, orders, history, memory,
              commission, slippage_steps, slippage_pct, stepsize, kline, verbose):
        """Step forward.
        """
        if verbose > 0:
            print(position['timestamp'], 'stepping', sep=':')
        previous_timestamp = position['timestamp']
        position['index'] += 1
        position['timestamp'] = timestamps[position['index']]
        timestamp = position['timestamp']
        periodic[timestamp] = periodic[previous_timestamp].copy()
        Backtest._book_order(order=order, timestamp=timestamp, orders=orders,
                            commission=commission, slippage_steps=slippage_steps, verbose=0)
        Backtest._fill_order(orders=orders, timestamp=timestamp, periodic=periodic,
                            kline=kline, commission=commission, slippage_pct=slippage_pct,
                            stepsize=stepsize, verbose=0)
        periodic[timestamp]['value'] = (periodic[timestamp]['cash'] +
                                        periodic[timestamp]['assets'] *
                                        kline[timestamp]['close'])
        history[0] = history[0][-memory+1:] + [list(periodic[timestamp].values())]

    @staticmethod
    def _book_order(order, timestamp, orders, commission, slippage_steps, verbose=0):
        """Book order.
        """
        if order == 0:
            return None
        if verbose > 0:
            print(timestamp, 'booking action', sep=':')
        desired_size = abs(order)
        action = int(np.sign(order))
        timestamp_filling = timestamp + timedelta(hours=slippage_steps)
        filled = False
        price = size = value = None
        action = ['sell', 'hold', 'buy'][action+1]
        values = {'dtm_booking':timestamp, 'filled':filled, 'action':action, 'price':price,
                  'desired_size':desired_size, 'size':size, 'commission':commission, 'value':value}
        orders[timestamp_filling] = values
        return None

    @staticmethod
    def _fill_order(orders, timestamp, periodic, kline, commission, slippage_pct, stepsize, verbose=0):
        """Fill order.
        """
        try:
            order = orders[timestamp]
        except KeyError:
            return None
        if verbose > 0:
            print(timestamp, 'filling action', sep=':')
        fkline = kline[timestamp]
        cash = periodic[timestamp]['cash']
        assets = periodic[timestamp]['assets']
        delta_cash = delta_assets = 0
        if order['action'] == 'buy':
            slippage_reference = fkline['high']
            price = fkline['close'] + (1 * slippage_pct * abs(slippage_reference-fkline['close']))
            precision = get_precision(stepsize)
            max_size = prec_floor(cash / (price * (1 + commission)), precision=precision)
            size = min(max_size, order['desired_size'])
            delta_assets = size
            value = size * price
            delta_cash = (value * (1 + commission)) * -1
        elif order['action'] == 'sell':
            slippage_reference = fkline['low']
            price = fkline['close'] + (-1 * slippage_pct * abs(slippage_reference-fkline['close']))
            size = min(assets, order['desired_size'])
            delta_assets = size * -1
            value = size * price
            delta_cash = (value * (1-commission))
        periodic[timestamp]['cash'] += delta_cash
        periodic[timestamp]['assets'] += delta_assets
        orders[timestamp]['filled'] = True
        orders[timestamp]['price'] = price
        orders[timestamp]['size'] = size
        orders[timestamp]['value'] = value
        return None

    def from_kline(self, kline, start_index=0):
        """From kline.
        """
        self.start_index = start_index
        self.index_max = len(kline)
        self.timestamps = kline.index
        self.position = {'index': self.start_index, 'timestamp':self.timestamps[self.start_index]}
        self.kline = kline.to_dict(orient='index')
        self.minPrice = kline.info.minPrice
        self.maxnPrice = kline.info.maxPrice
        self.tickSize = kline.info.tickSize
        self.minQty = kline.info.minQty
        self.maxQty = kline.info.maxQty
        self.stepSize = kline.info.stepSize
        self._orders = dict()
        self._periodic = dict()
        self._periodic[self.position['timestamp']] = {'cash':self.init_cash, 'assets':self.init_assets,
                                                      'value':(self.init_cash+
                                                               self.init_assets*
                                                               self.kline[self.position['timestamp']]['close'])}
        self._history = {0:[[0, 0, 0] for _ in range(self.memory-1)] +
                           [list(self._periodic[self.position['timestamp']].values())]}

    def run(self, orders, verbose=0):
        """Step forward multiple times.
        """
        for order in orders:
            self.step(order=order, verbose=verbose)
        return None

    def step(self, order, verbose=0):
        """Step forward.
        """
        Backtest._step(order=order, position=self.position, timestamps=self.timestamps,
                       periodic=self._periodic, orders=self._orders, history=self._history,
                       memory=self.memory, commission=self.commission,
                       slippage_steps=self.slippage_steps, slippage_pct=self.slippage_pct,
                       stepsize=self.stepSize, kline=self.kline, verbose=verbose)

    def stepx(self, n=1, orders=None, inplace=False, verbose=0):
        """Step forward while catching result.
        """
        orders = np.repeat(0, n) if orders is None else orders
        if not inplace:
            position = self.position.copy()
            _periodic = self._periodic.copy()
            _orders = self._orders.copy()
            _history = self._history.copy()
            for order in orders:
                Backtest._step(order=order, position=position, timestamps=self.timestamps,
                               periodic=_periodic, orders=_orders, history=_history,
                               memory=self.memory, commission=self.commission,
                               slippage_steps=self.slippage_steps, slippage_pct=self.slippage_pct,
                               stepsize=self.stepSize, kline=self.kline, verbose=verbose)
            return _periodic, _orders, _history

class SingleAssetEnv(gym.Env):
    """Single Asset Environment.
    """
    _rel_indicators_stem = ('sma', 'ema', 'wma', 'bb', 'dc', 'kc', 'psar', 'vwap')
    _abs_indicators_stem = ('macd', 'adx', 'rsi', 'atr', 'cmf', 'mfi', 'roc', 'stoch', 'd_ret', 'd_logret')

    def __init__(self, klmngr, assets, backtest=None, window=24, datetimes=None, randomize_start=True, allow_gaps=False,
                 episode_steps=-1):
        super().__init__()

        self.action_space = gym.spaces.Discrete(3)
        self.obs_cols = klmngr[klmngr.assets[0]].indicators.columns.size
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(window*(self.obs_cols+1),),
                                                dtype=np.float32)
        self.klmngr = klmngr
        self.assets = assets
        self.backtest = Backtest() if backtest is None else backtest
        self.window = window
        self.datetimes =  SingleAssetEnv.get_datetimes(datetimes)
        self.randomize_start = randomize_start
        self.allow_gaps = allow_gaps
        self.episode_steps = episode_steps
        self.reset()

    @staticmethod
    def get_datetimes(datetimes):
        """Get datetimes.
        """
        if datetimes is None:
            datetimes =  (pd.Timestamp(1970, 1, 1), pd.Timestamp(2050, 1, 1))
        elif isinstance(datetimes, tuple):
            pass
        else:
            datetimes =  (pd.Timestamp(1970, 1, 1), datetimes)
        return datetimes

    def render(self):
        """Render stuff.
        """
        return None

    def get_observation(self, timestamp, window):
        """Get observation.
        """
        # timestamp = timestamp + pd.Timedelta(days=1)
        close_values = self.kline.loc[:timestamp, :].close.values[-window:].reshape(-1, 1)
        abs_indicator_values = self.kline.indicators.loc[:timestamp, self.abs_indicators].fillna(-1).values[-window:]
        rel_indicator_values = self.kline.indicators.loc[:timestamp, self.rel_indicators].fillna(-1).values[-window:]
        observation = np.hstack((abs_indicator_values, rel_indicator_values / close_values))
        periodic_values = np.array(self.backtest.history)[:, :2]
        transactions_history = np.hstack([0, 0] +
                                         [(periodic_values[:-1, _] != periodic_values[1:, _]).astype(int)
                                          for _ in [0, 1]])
        return np.append(observation.T.flatten(), transactions_history)

    def get_observation_old(self, timestamp, window):
        """Get observation.
        """
        # timestamp = timestamp + pd.Timedelta(days=1)
        close_values = self.kline.loc[:timestamp, :].close.values[-window:].reshape(-1, 1)
        # indicator_values = self.kline.indicators.loc[:timestamp, :].fillna(-1).values[-window:]
        periodic_values = np.array(self.backtest.history)[:, :2]
        observation_raw = np.hstack((close_values, )) # indicator_values # periodic_values
        if self.current_step % 720 == 0: # 720 == 24 * 30 # self.current_step == 0:
            scaler = StandardScaler() #MinMaxScaler((-1, 1))
            # Needs:
            # - Observation data which is similarly scaled accross assets
            # - Observation data which is similarly scaled accross periods
            # - Robust to not only historic outliers but also to future outliers (see options 4)
            # Options:
            # 1) Fit on all historic data and transform observations
            # 2) Fit on pre-episode historic data and transform observations
            # 3) Fit on subset (e.g. 200 ticks) of pre-episode historic data and transform observations
            # 4) Fit on 3, 2 or 1 + forecasted data and transform observations
            scaler_history = 5000
            hist_close_values = self.kline.loc[:timestamp, :].close.values[-scaler_history:].reshape(-1, 1)
            '''
            hist_indicator_values = self.kline.indicators.loc[:timestamp, :].fillna(-1).values[-scaler_history:]
            hist_periodic_values = self.backtest.periodic.loc[:timestamp, :].values[-scaler_history:, :2]
            hist_periodic_values = hist_periodic_values[~(hist_periodic_values==0).all(axis=1)]
            if hist_periodic_values.shape[0] != hist_indicator_values.shape[0]:
                # try:
                #     stepx = self.backtest.stepx(orders=np.random.randint(-10, 10, size=10))[0]
                # except:
                #     stepx = self.backtest.stepx(orders=np.random.randint(-10, 10, size=1))[0]
                # rand_periodic_values = pd.DataFrame(stepx).T.values
                # shape = hist_indicator_values.shape[0]
                # hist_periodic_values_rep = np.vstack([periodic_values]*(int(shape/2/periodic_values.shape[0])+1))
                # rand_periodic_values_rep = np.vstack([rand_periodic_values]*(int(shape/2/10)+1))
                # hist_periodic_values = np.vstack([hist_periodic_values_rep, rand_periodic_values_rep])[-shape:]
                shape_0 = hist_indicator_values.shape[0]
                # hist_periodic_values = np.vstack([periodic_values]*(int(shape/periodic_values.shape[0])+1))[-shape:]
                nans = np.full((shape_0, 2), np.nan)
                hist_periodic_values = np.vstack([nans, hist_periodic_values])[-shape_0:]
            hist_observation_raw = np.hstack((hist_close_values, hist_indicator_values, hist_periodic_values))
            '''
            fitted_scaler = scaler.fit(hist_close_values)
            setattr(self, 'scaler', fitted_scaler)
        # self.scaler.transform(observation_raw).T.ravel()
        observation = np.hstack([self.scaler.fit_transform(observation_raw[:, [_]]) for _ in range(self.obs_cols)])
        # nr_of_transactions = (-1+2*sum(periodic_values[:-1, 1] != periodic_values[1:, 1])/(window-1)).reshape(-1, 1)
        transactions_history = np.hstack([0, 0] +
                                         [(periodic_values[:-1, _] != periodic_values[1:, _]).astype(int)
                                          for _ in [0, 1]])
        return np.append(observation.T.flatten(), transactions_history)

    def get_reward(self):
        """Get reward.
        """
        # future_periodic = pd.DataFrame(self.backtest.stepx(n=1)[0]).T
        timestamp = self.backtest.position['timestamp']
        previous_timestamp = timestamp - timedelta(hours=1)
        while True:
            try:
                value_t0 = getattr(self.backtest, '_periodic')[previous_timestamp]['value']
                price_t0 = getattr(self.kline, 'close')[previous_timestamp]
                break
            except KeyError:
                previous_timestamp -= timedelta(hours=1)
        value_t1 = getattr(self.backtest, '_periodic')[timestamp]['value']
        price_t1 = getattr(self.kline, 'close')[timestamp]
        # reward = (value_t1-value_t0)/value_t0
        # reward = reward-1 if reward <= 0 else 1+reward
        # return reward
        # value_d = ((value_t1-value_t0)/value_t0)
        # price_d = ((price_t1-price_t0)/price_t0)
        # return value_d / price_d
        amount_t0 = value_t0 / price_t0
        return (value_t1 - value_t0) - (amount_t0*price_t1 - amount_t0*price_t0)


    def get_info(self):
        """Get info.
        """
        timestamp = self.backtest.position['timestamp']
        info = {'periodic': {timestamp: getattr(self.backtest, '_periodic')[timestamp]}}
        return info

    def reset(self):
        """Reset.
        """
        while True:
            asset = np.random.choice(self.assets)
            kline = self.klmngr[asset].copy()
            index_to_drop = kline.index[(kline.index < self.datetimes[0]) | (kline.index >= self.datetimes[1])]
            kline.drop(index_to_drop, inplace=True)
            if kline.shape[0] > self.window*5:
                break
        if self.randomize_start:
            while True:
                index = np.random.choice(np.arange(kline.shape[0]-5))
                timestamp = kline.index[index]
                if kline.index[:index].size > self.window:
                    break
        else:
            index = 0+self.window
            timestamp = kline.index[index]
        self.current_step = 0
        self.scaler = None
        self.kline = kline
        self.backtest.from_kline(kline, start_index=index)
        self.rel_indicators = self.kline.indicators.columns.to_series().str.startswith(self._rel_indicators_stem)
        self.abs_indicators = self.kline.indicators.columns.to_series().str.startswith(self._abs_indicators_stem)
        observation = self.get_observation(timestamp=timestamp, window=self.window)
        return observation

    def step(self, action):
        """Step forward.
        """
        assert(action in [0, 1, 2])
        self.current_step += 1
        self.backtest.step(order=(action - 1) * 999)
        timestamp = self.backtest.position['timestamp']
        observation = self.get_observation(timestamp = timestamp,
                                           window=self.window)
        reward = self.get_reward()
        done = (timestamp == self.kline.index.max()-timedelta(hours=1) or
                (not(self.allow_gaps) and
                 any([timestamp + timedelta(hours=1) not in self.kline.index,
                      timestamp + timedelta(hours=2) not in self.kline.index])) or
                (self.current_step == self.episode_steps))
        info = self.get_info()
        return observation, reward, done, info

    def close(self):
        """Close.
        """
        return None
