# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import gzip
import os
import pathlib
import pickle
import shutil
import socket
import time

import yaml
import petname
import numpy as np
import pandas as pd
import torch

from train_test import test, estimate
from data import perturb
from utils import (
    initialize_logging,
    nploadp,
    import_string,
    get_free_gpu_ids,
    send_twilio_message
)

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False


def configure_logging(app_name: str, run_id: str = None, logdir: str = "logs", debug=False):
    """Set up logging to both file and console."""
    debugtag = "-debug" if debug else ""
    run_id = str(run_id)
    username = os.path.split(os.path.expanduser("~"))[-1]
    hostname = socket.gethostname()
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    starttimestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logtag = petname.Generate(2)

    fh = logging.FileHandler(
        f"{logdir}/{app_name}{debugtag}_{run_id}_{logtag}_{username}_{hostname}_{starttimestr}.log"
    )
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        f"[%(asctime)s] Run-{run_id} - %(levelname)s - %(message)s", '%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logging.getLogger('').handlers = []
    logging.getLogger('').addHandler(fh)
    logging.getLogger('').addHandler(ch)
    logging.info(f"STARTED LOGGING FROM CHILD")
    return username, hostname, logtag, starttimestr


def run(config: dict,
        run_id: str = None,
        gpu_device_ids: list = None,
        notification_phone_number: str = None,):
    """
    Run test/estimate of CNNTransformer on crypto residuals.
    """
    model_name = config['model_name']
    results_tag = config['results_tag']
    debug = config['debug']
    username, hostname, log_tag, starttime = configure_logging(model_name, run_id=run_id, debug=debug) \
        if run_id else initialize_logging(model_name, debug=debug)

    try:
        logging.info(f"Config: \n{json.dumps(config, indent=2, sort_keys=False)}")

        results_filename = f"results_{log_tag}_{results_tag}"
        factor_models = config['factor_models']
        cap = config['cap_proportion']
        objective = config['objective']

        filepaths = []
        residual_weightsNames = []
        datanames = []
        results_dict = {}

        # === Crypto PCA filepaths ===
        pcadir = "crypto"
        pcartag = "AvPCA_OOSresiduals"
        pcamtag = "AvPCA_OOSMatrixresiduals"
        for factor in factor_models.get("PCA", []):
            ioy, w, cw = 2020, 60, 252  # crypto naming convention
            filepaths += [f"residuals/{pcadir}/{pcartag}_{factor}_factors_{ioy}_initialOOSYear_{w}_rollingWindow_{cw}_covWindow_{cap}_Cap.npy"]
            datanames += [f'CryptoPCA{factor}']
            residual_weightsNames += [f"residuals/{pcadir}/{pcamtag}_{factor}_factors_{ioy}_initialOOSYear_{w}_rollingWindow_{cw}_covWindow_{cap}_Cap.npy"]

        # === Dates for crypto ===
        daily_dates = pd.date_range(start="2020-01-01", end="2025-01-01", freq="D")

        # === Main loop ===
        for i, filepath in enumerate(filepaths):
            logging.info(f'Testing {filepath}')
            if not os.path.exists(filepath) and os.path.exists(filepath + ".gz"):
                logging.info("Unzipping residual file")
                with gzip.open(filepath + ".gz", 'rb') as f_in:
                    with open(filepath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            residuals = np.load(filepath).astype(np.float32)
            if 'perturbation' in config and config['perturbation']:
                logging.info(f"Before perturbing residuals: std: {np.std(residuals[residuals != 0]):0.4f}")
                residuals = perturb(residuals, config['perturbation'])
                logging.info(f"After perturbing residuals: std: {np.std(residuals[residuals != 0]):0.4f}")
            residuals[np.isnan(residuals)] = 0

            if config.get('use_residual_weights', False):
                logging.info('Loading residual composition matrix')
                residual_weights = nploadp(residual_weightsNames[i])
                logging.info('Residual composition matrix loaded')
                residual_weight_marker = "__residual_weights"
            else:
                residual_weights = None
                residual_weight_marker = ""

            if objective not in ["sharpe", "meanvar", "sqrtMeanSharpe"]:
                raise Exception(f"Invalid objective '{objective}'")

            # === Model & preprocess ===
            model = import_string(f"models.{config['model_name']}.{config['model_name']}")
            preprocess = import_string(f"preprocess.{config['preprocess_func']}")

            model_tag = datanames[i] + \
                        f"__{config['model_name']}{residual_weight_marker}__{objective}" + \
                        f"__{config['trans_cost']}trans_cost__{config['hold_cost']}hold_cost" + \
                        f"__{config['model']['lookback']}lookback__{config['length_training']}length_training" + \
                        (f"__{results_tag}" if results_tag else "") + f"__{log_tag}"

            logging.info('STARTING: ' + model_tag)

            # === Device assignment ===
            if gpu_device_ids is None:
                try:
                    if torch.cuda.is_available():
                        device_ids = get_free_gpu_ids(min_memory_mb=9000)
                        if not device_ids:
                            logging.warning("No free GPU found → using CPU")
                            device_ids = []
                    else:
                        logging.warning("CUDA not available → using CPU")
                        device_ids = []
                except Exception:
                    logging.warning("GPU detection failed → using CPU")
                    device_ids = []
            else:
                device_ids = gpu_device_ids

            device_str = f'cuda:{device_ids[0]}' if device_ids else 'cpu'

            outdir = os.path.join(str(pathlib.Path().resolve()), 'results', config['model_name'])
            os.makedirs(outdir, exist_ok=True)

            # === Run test/estimate ===
            if config['mode'] == 'test':
                rets, sharpe, ret, std, turnover, short_proportion = test(
                    residuals, daily_dates, model, preprocess, config,
                    residual_weights=residual_weights, save_params=True,
                    force_retrain=config['force_retrain'], parallelize=True,
                    log_dev_progress_freq=10, log_plot_freq=149,
                    device=device_str, device_ids=device_ids,
                    output_path=outdir, num_epochs=config['num_epochs'],
                    early_stopping=config['early_stopping'], model_tag=model_tag,
                    batchsize=config['batch_size'], retrain_freq=config['retrain_freq'],
                    rolling_retrain=config['rolling_retrain'], length_training=config['length_training'],
                    lookback=config['model']['lookback'], trans_cost=config['trans_cost'],
                    hold_cost=config['hold_cost'], objective=config['objective']
                )
            elif config['mode'] == 'estimate':
                rets, sharpe, ret, std, turnover, short_proportion = estimate(
                    residuals, daily_dates, model, preprocess, config,
                    residual_weights=residual_weights, save_params=True,
                    force_retrain=config['force_retrain'], parallelize=True,
                    log_dev_progress_freq=10, log_plot_freq=149,
                    device=device_str, device_ids=device_ids,
                    output_path=outdir, num_epochs=config['num_epochs'],
                    lr=config['learning_rate'], early_stopping=config['early_stopping'],
                    model_tag=model_tag, batchsize=config['batch_size'],
                    length_training=config['length_training'], test_size=config['retrain_freq'],
                    lookback=config['model']['lookback'], trans_cost=config['trans_cost'],
                    hold_cost=config['hold_cost'], objective=config['objective']
                )
            else:
                raise Exception(f"Invalid mode '{config['mode']}'")

            results_dict[model_tag] = {
                "returns": rets,
                "sharpe": sharpe,
                "ret": ret,
                "std": std,
                "turnover": turnover,
                "short_proportion": short_proportion,
                "config": config,
                "timestamp": datetime.datetime.now()
            }

            pkl_filename = f'results/{model_name}/{results_filename}.pickle'
            if os.path.exists(pkl_filename):
                pkl_filename = pkl_filename.replace(".pickle", f"_{int(time.time())}.pickle")
            with open(pkl_filename, 'wb') as handle:
                pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        logging.error("Uncaught exception", exc_info=e)
        if notification_phone_number:
            error_msg = f"FAILED: {results_tag} - {model_name} - {log_tag} - {hostname} - {starttime} - {repr(e)}"
            send_twilio_message(error_msg, notification_phone_number)
        raise e

    if notification_phone_number:
        completion_msg = f"COMPLETED: {results_tag} - {model_name} - {log_tag} - {hostname} - {starttime}"
        send_twilio_message(completion_msg, notification_phone_number)


def init_argparse():
    parser = argparse.ArgumentParser(
        description="Test trading policy model on crypto residual time series given configuration file."
    )
    parser.add_argument("--config", "-c", help="path to a .yaml configuration file", required=True)
    parser.add_argument("--run-id", "-r", help="identifier string (e.g. 'run42')", required=False)
    parser.add_argument("--gpu-device-ids", "-g", nargs="*", type=int, help="GPU device IDs to use", required=False)
    parser.add_argument("--notification-phone-number", "-p", help="phone number for SMS alerts", required=False)
    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            args.config = config
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    print("Running...")
    run(**vars(args))


if __name__ == "__main__":
    main()
