import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression


class CryptoPCA:
    def __init__(self, tickers, start="2020-01-01", end="2025-01-01",
                 logdir="/content/drive/MyDrive/DLSA/dlsa-public/residuals/crypto"):
        """
        PCA residual generator for crypto assets.

        Args:
            tickers: list of crypto tickers (e.g., ["BTC-USD","ETH-USD","SOL-USD"])
            start: start date for data
            end: end date for data
            logdir: directory to save residual npy files
        """
        # === Download price data ===
        data = yf.download(tickers, start=start, end=end)["Close"]
        data = data.dropna(how="all")

        # === Daily log returns ===
        returns = np.log(data).diff().dropna()
        self.dailyData = returns.values  # T x N
        self.dailyDates = returns.index
        self.monthlyDates = pd.date_range(start=returns.index.min(),
                                          end=returns.index.max(), freq="M")

        # === Dummy monthly data / caps (to mimic equity pipeline) ===
        Tm, N = len(self.monthlyDates), self.dailyData.shape[1]
        self.monthlyData = np.zeros((Tm, N, 1))  # placeholder
        self.monthlyDataUnnormalized = np.zeros((Tm, N, 20))
        self.monthlyCaps = np.ones((Tm, N))  # equal weights for all coins

        # Save location
        self._logdir = logdir
        os.makedirs(self._logdir, exist_ok=True)

        self.tickers = tickers

    def OOSRollingWindowPermnos(self, save=True, printOnConsole=True,
                                initialOOSYear=2020,
                                sizeWindow=30,  # reduced for crypto
                                sizeCovarianceWindow=90,  # reduced for crypto
                                CapProportion=0.01,
                                factorList=range(0, 5)):

        Rdaily = self.dailyData.copy()
        T, N = Rdaily.shape
        firstOOSDailyIdx = np.argmax(self.dailyDates.year >= initialOOSYear)
        firstOOSMonthlyIdx = np.argmax(self.monthlyDates.year >= initialOOSYear)

        # === All assets considered (at least 30 valid obs after start) ===
        assetsToConsider = (np.count_nonzero(~np.isnan(Rdaily[firstOOSDailyIdx:, :]), axis=0) >= 30)
        Ntilde = np.sum(assetsToConsider)
        print('N', N, 'Ntilde', Ntilde)

        for factor in factorList:
            residualsOOS = np.zeros((T - firstOOSDailyIdx, N), dtype=float)
            residualsMatricesOOS = np.zeros((T - firstOOSDailyIdx, Ntilde, Ntilde), dtype=np.float32)

            for t in range(T - firstOOSDailyIdx):
                # Check if enough history exists
                if (t + firstOOSDailyIdx) < sizeCovarianceWindow:
                    continue

                idxsSelected = ~np.any(
                    Rdaily[(t + firstOOSDailyIdx - sizeCovarianceWindow + 1):(t + firstOOSDailyIdx + 1), :] == 0,
                    axis=0).ravel()

                if np.sum(idxsSelected) == 0:
                    continue  # skip if no assets selected

                if factor == 0:
                    residualsOOS[t:(t + 1), idxsSelected] = Rdaily[
                        (t + firstOOSDailyIdx):(t + firstOOSDailyIdx + 1), idxsSelected
                    ]
                else:
                    res_cov_window = Rdaily[
                        (t + firstOOSDailyIdx - sizeCovarianceWindow):(t + firstOOSDailyIdx), idxsSelected
                    ]
                    if res_cov_window.shape[0] < sizeWindow:
                        continue  # not enough rows

                    res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                    res_vol = np.sqrt(np.mean((res_cov_window - res_mean) ** 2, axis=0, keepdims=True))
                    res_vol[res_vol == 0] = 1.0  # prevent division by zero
                    res_normalized = (res_cov_window - res_mean) / res_vol

                    Corr = np.dot(res_normalized.T, res_normalized)
                    eigenValues, eigenVectors = np.linalg.eigh(Corr)
                    loadings = eigenVectors[:, -factor:].real

                    factors = np.dot(res_cov_window[-sizeWindow:, :] / res_vol, loadings)
                    if factors.shape[0] == 0:
                        continue

                    old_loadings = loadings
                    regr = LinearRegression(fit_intercept=False, n_jobs=-1).fit(
                        factors, res_cov_window[-sizeWindow:, :]
                    )
                    loadings = regr.coef_

                    DayFactors = np.dot(Rdaily[t + firstOOSDailyIdx, idxsSelected] / res_vol, old_loadings)
                    residuals = Rdaily[t + firstOOSDailyIdx, idxsSelected] - DayFactors.dot(loadings.T)
                    residualsOOS[t:(t + 1), idxsSelected] = residuals

                    MatrixFull = np.zeros((N, N))
                    MatrixReduced = (np.eye(len(res_cov_window[-1, :]))
                                     - np.diag(1 / res_vol.squeeze()) @ old_loadings @ loadings.T)
                    idxsSelected2 = idxsSelected.reshape((N, 1)) @ idxsSelected.reshape((1, N))
                    MatrixFull[idxsSelected2] = MatrixReduced.ravel()
                    residualsMatricesOOS[t:(t + 1)] = MatrixFull[assetsToConsider][:, assetsToConsider].T

                if t % 50 == 0 and printOnConsole:
                    print(f"At date {self.dailyDates[t+firstOOSDailyIdx]}, "
                          f"factor={factor}, selected={np.sum(idxsSelected)}")

            np.nan_to_num(residualsOOS, copy=False)
            np.nan_to_num(residualsMatricesOOS, copy=False)

            if save:
              res_name = (f"AvPCA_OOSresiduals_{factor}_factors_{initialOOSYear}_initialOOSYear_"
                          f"{sizeWindow}_rollingWindow_{sizeCovarianceWindow}_covWindow_{CapProportion}_Cap.npy")
              mat_name = (f"AvPCA_OOSMatrixresiduals_{factor}_factors_{initialOOSYear}_initialOOSYear_"
                          f"{sizeWindow}_rollingWindow_{sizeCovarianceWindow}_covWindow_{CapProportion}_Cap.npy")

              np.save(os.path.join(self._logdir, res_name), residualsOOS)
              np.save(os.path.join(self._logdir, mat_name), residualsMatricesOOS)
              print(f"✅ Saved {res_name} and {mat_name}")
