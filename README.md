# Deep Learning in Cryptocurrencies

This repository contains my own implementation of deep learning models for cryptocurrency prediction and analysis.

## Inspiration
This work was inspired by the open-source project [dlsa-public](https://github.com/gregzanotti/dlsa-public).  
Building on that foundation, I extended the approach by implementing a **CNN + Transformer hybrid model** for cryptocurrencies.

## My Contributions
- Implemented a **CNN + Transformer architecture** to capture both local patterns and long-range dependencies in crypto time-series data.
- Cleaned up legacy code and simplified the repo structure.
- Added reproducible configs and training scripts for cryptocurrency experiments.

## Installation

pip install petname guinfo twilio

## Run experiments

python -c "import factor_models; factor_models.run_crypto_pca()"
python run_train_test_crypto.py --config configs/crypto-full.yaml

