The cryptoast library provides functionalities for storing, analyzing and taking action on crypto currency exchange market information.

Api-s used are:

-   BINANCE for data retrieval
-   BINANCE for order execution
-   AWS for data storage
-   AWS for model management
-   AWS for order initiation. 


Main functionalities
====================

A number of classes are available.
The Examples notebook shows off some of the library's functionalities.

Kline
-----

Object for asset-specific data storage, retrieval, and analysis.

Klines
------

Object for regrouping multiple Kline objects, and representing them as a readable list of official asset acronyms.

KLMngr
------

Management object for all available assets.

KLModel
-------

Reinforcement Learning investment decision maker.
