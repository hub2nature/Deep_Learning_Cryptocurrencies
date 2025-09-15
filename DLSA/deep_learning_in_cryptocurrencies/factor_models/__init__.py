from factor_models.cpca import CryptoPCA

def run_crypto_pca():
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
    cpca = CryptoPCA(tickers, start="2020-01-01", end="2025-01-01")
    cpca.OOSRollingWindowPermnos(
        initialOOSYear=2020,
        sizeWindow=60,
        sizeCovarianceWindow=252,
        CapProportion=0.01,
        factorList=[0, 1, 2, 3]
    )
