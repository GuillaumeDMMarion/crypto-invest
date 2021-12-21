from setuptools import setup

setup(
    name="cryptoast",
    version="0.1",
    author="Guillaume Marion",
    packages=["cryptoast"],
    url="http://github.com/GuillaumeDMMarion/crypto-invest",
    description="Library for managing crypto currency assets",
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
        "tqdm",
        "requests",
        "python-binance",
        "pip",
    ],
)
