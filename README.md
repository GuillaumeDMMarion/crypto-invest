The cryptoast library provides functionalities for retrieving, storing, and analyzing data. Potenially it also provides functionalities for taking actions based on this data; Either through inferred signals' thresholds or by using a reinforcement learning layer on top of it.

&nbsp;

Api(s) used are:

-   **BINANCE** for data retrieval and order execution
-   **AWS** for data storage (optional, local storage can be used)

&nbsp;

Main prerequisites
==================
- pandas
- numpy
- gym
- ta
- s3fs
- tensorflow
- scikit-learn
- python-binance
- stable-baselines

&nbsp;

Main functionalities
====================

A number of packages and subsequent modules are available:

&nbsp;

base.kline.Kline
----------------

Object for asset-specific data storage, retrieval, and analysis.

&nbsp;

base.klines.Klines
------------------

Object for regrouping multiple Kline objects, and representing them as a readable list of official asset acronyms.

&nbsp;

base.klines.KLMngr
------------------

Management object for all available assets.

&nbsp;

model.agents.Backtest
---------------------

In-house backtester, agent-feedable.

&nbsp;

model.agents.SingleAssetEnv
---------------------------

Single Asset Environment for reinforcement-learning purposes.

&nbsp;

Installation
============

```bash
pip install -e cryptoast
```


&nbsp;

Minimal example
===============

Local storage update and average signal example.

```python
from binance.client import Client
from cryptoast.base.kline import Kline

client = Client('key', 'secret')
kline = Kline('ETHUSDT')
kline.update(client, store=True)
if kline.signals.iloc[-1, :].eq(1).mean() > .5:
    print('buy')
```

&nbsp;

Notebook examples
=================

implementation_rl.ipynb
-----------------------
Most rubble-pruned notebook showing mainly: initialization, data update, dummy buy-sell signal computations and rl implementation tryouts.

&nbsp;

Exploration.ipynb
-----------------------
Exhaustive testing notebook showing amongst other: initialization, data update, single-asset buy-sell signal computation tryouts, multi-asset buy-sell signal computation tryouts, rl implementation tryouts, categorical modeling tryouts, etc.

&nbsp;

Examples.ipynb
-----------------------
Old notebook showing minimal functionalities.

&nbsp;

Disclaimer
==========
No relation whatsoever to http://cryptoast.fr