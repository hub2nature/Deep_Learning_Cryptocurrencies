# -*- coding: utf-8 -*-
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import import_string

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False

EPS = 1e-8  # numerical safety epsilon


def _to_torch(x, device):
    """Create a torch tensor on the right device without breaking autograd for model params."""
    return torch.as_tensor(x, device=device)


def train(model,
          preprocess,
          data_train,
          data_dev=None,
          log_dev_progress=True, log_dev_progress_freq=50, log_plot_freq=50,
          num_epochs=100, lr=0.001, batchsize=200,
          optimizer_name="Adam", optimizer_opts={"lr": 0.001},
          early_stopping=False, early_stopping_max_trials=5, lr_decay=0.5,
          residual_weights_train=None, residual_weights_dev=None,
          save_params=True, output_path=None, model_tag='',
          lookback=30,
          trans_cost=0, hold_cost=0,
          parallelize=True, device=None, device_ids=[0,1,2,3,4,5,6,7],
          force_retrain=True,
          objective="sharpe"):

    if output_path is None:
        output_path = model.logdir
    if device is None:
        device = model.device
    logging.info(f"train(): data_train.shape {data_train.shape}")

    # pick assets with at least `lookback` nonzeros in sample
    assets_to_trade = np.count_nonzero(data_train, axis=0) >= lookback
    logging.info(f"train(): assets_to_trade.shape {assets_to_trade.shape}")
    data_train = data_train[:, assets_to_trade]
    if residual_weights_train is not None:
        residual_weights_train = residual_weights_train[:, assets_to_trade]
    T, N = data_train.shape
    logging.info(f"train(): T {T} N {N}")

    windows, idxs_selected = preprocess(data_train, lookback)
    idxs_selected = torch.as_tensor(idxs_selected, device=device)

    # Sanity: how many valid windows total?
    total_valid = int(torch.count_nonzero(idxs_selected).item())
    logging.info(f"train(): windows.shape {windows.shape} idxs_selected.shape {idxs_selected.shape} "
                 f"total_valid={total_valid}")
    if total_valid == 0:
        raise ValueError("No valid positions found in training data — check lookback or residual coverage.")

    # model / optimizer
    if parallelize:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.train()
    optimizer_func = import_string(f"torch.optim.{optimizer_name}")
    optimizer = optimizer_func(model.parameters(), **optimizer_opts)

    begin_time = time.time()

    for epoch in range(num_epochs):
        rets_full = np.zeros(T - lookback, dtype=np.float32)
        short_proportion = np.zeros(T - lookback, dtype=np.float32)
        turnover = np.zeros(T - lookback, dtype=np.float32)

        num_batches = int((T - lookback) / batchsize) + 1
        for i in range(num_batches):
            lo = batchsize * i
            hi = min(batchsize * (i + 1), T - lookback)
            if hi <= lo:
                continue

            idxs_batch = idxs_selected[lo:hi, :]  # (B, N)
            valid_count = int(torch.count_nonzero(idxs_batch).item())
            if epoch == 0 and i == 0:
                logging.info(f"epoch {epoch} batch {i} B={hi-lo} N={N} valid_positions={valid_count}")

            if valid_count == 0:
                logging.debug(f"Skipping batch {i} (no valid positions)")
                continue

            inputs = windows[lo:hi][idxs_batch.cpu().numpy()]  # (num_valid, lookback)
            inputs_t = _to_torch(inputs, device)
            out = model(inputs_t).reshape(-1)

            # DIFFERENTIABLE scatter back to dense weight matrix
            weights = torch.zeros((hi - lo, N), device=device)
            weights = weights.masked_scatter(idxs_batch, out)

            # normalize weights safely
            abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True)
            abs_sum = torch.clamp(abs_sum, min=EPS)
            weights = weights / abs_sum

            # compute batch returns
            y_batch = _to_torch(data_train[lookback + lo:lookback + hi, :], device)
            rets_train = torch.sum(weights * y_batch, dim=1)

            # frictions
            if hi - lo > 1:
                tcost = torch.sum(torch.abs(weights[1:] - weights[:-1]), dim=1)
                tcost = torch.cat((torch.zeros(1, device=device), tcost))
            else:
                tcost = torch.zeros(1, device=device)
            hcost = torch.sum(torch.abs(torch.minimum(weights, torch.zeros(1, device=device))), dim=1)
            rets_train = rets_train - trans_cost * tcost - hold_cost * hcost

            # loss
            mean_ret = torch.mean(rets_train)
            std = torch.clamp(torch.std(rets_train), min=EPS)

            assert mean_ret.requires_grad, "mean_ret lost grad tracking"
            assert std.requires_grad, "std lost grad tracking"

            if objective == "sharpe":
                loss = -mean_ret / std
            elif objective == "meanvar":
                loss = -mean_ret * 252.0 + std * 15.9
            elif objective == "sqrtMeanSharpe":
                loss = -torch.sign(mean_ret) * torch.sqrt(torch.abs(mean_ret) + EPS) / std
            else:
                raise Exception(f"Invalid objective loss {objective}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # numpy for logs
            rets_full[lo:hi] = rets_train.detach().cpu().numpy()
            w_np = weights.detach().cpu().numpy()
            if hi - lo > 1:
                turnover[lo:hi - 1] = np.sum(np.abs(w_np[1:] - w_np[:-1]), axis=1)
                turnover[hi - 1] = turnover[hi - 2]
            else:
                turnover[lo:hi] = 0.0
            short_proportion[lo:hi] = np.sum(np.abs(np.minimum(w_np, 0.0)), axis=1)

        if log_dev_progress and epoch % log_dev_progress_freq == 0:
            full_ret = np.mean(rets_full)
            full_std = np.std(rets_full)
            full_sharpe = 0.0 if full_std <= 0 else full_ret / full_std
            full_turnover = np.mean(turnover)
            full_short_proportion = np.mean(short_proportion)

            logging.info(
                f"Epoch {epoch}/{num_epochs}, "
                f"train Sharpe {full_sharpe*np.sqrt(252):.2f}, "
                f"ret {full_ret*252:.4f}, "
                f"std {full_std*np.sqrt(252):.4f}, "
                f"turnover {full_turnover:.4f}, "
                f"short_prop {full_short_proportion:.4f}, "
                f"time/epoch {(time.time()-begin_time)/(epoch+1):.2f}s"
            )

    logging.info(f"Training done - Model: {model_tag}")
    return rets_full, turnover, short_proportion, w_np, assets_to_trade


def get_returns(model, preprocess, objective, data_test, lookback=30,
                trans_cost=0, hold_cost=0, residual_weights=None,
                device=None, parallelize=False, device_ids=[0]):
    """Run inference only (no gradient)."""
    if device is None:
        device = model.device
    if parallelize:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)

    assets_to_trade = np.count_nonzero(data_test, axis=0) >= lookback
    data_test = data_test[:, assets_to_trade]
    T, N = data_test.shape
    windows, idxs_selected = preprocess(data_test, lookback)
    idxs_selected = torch.as_tensor(idxs_selected, device=device)
    model.eval()

    with torch.no_grad():
        weights = torch.zeros((T - lookback, N), device=device)
        valid_count = int(torch.count_nonzero(idxs_selected).item())
        if valid_count > 0:
            out = model(_to_torch(windows[idxs_selected.cpu().numpy()], device)).reshape(-1)
            weights = weights.masked_scatter(idxs_selected, out)

        abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True)
        abs_sum = torch.clamp(abs_sum, min=EPS)
        weights = weights / abs_sum

        rets_test = torch.sum(weights * _to_torch(data_test[lookback:T, :], device), dim=1)
        tcost = torch.cat((torch.zeros(1, device=device),
                           torch.sum(torch.abs(weights[1:] - weights[:-1]), dim=1))) if T - lookback > 1 else torch.zeros(1, device=device)
        hcost = torch.sum(torch.abs(torch.minimum(weights, torch.zeros(1, device=device))), dim=1)
        rets_test = rets_test - trans_cost * tcost - hold_cost * hcost
        if tcost.numel() > 1:
            tcost[0] = torch.mean(tcost[1:])

        mean = torch.mean(rets_test)
        std = torch.clamp(torch.std(rets_test), min=EPS)

        if objective == "sharpe":
            loss = -mean / std
            sharpe = -loss
        elif objective == "meanvar":
            loss = -mean * 252 + std * 15.9
            sharpe = mean / std
        elif objective == "sqrtMeanSharpe":
            loss = -torch.sign(mean) * torch.sqrt(torch.abs(mean) + EPS) / std
            sharpe = mean / std
        else:
            raise Exception(f"Invalid objective loss {objective}")

    return (rets_test.cpu().numpy(), loss, sharpe.cpu().numpy(),
            tcost.cpu().numpy(), hcost.cpu().numpy(),
            weights.cpu().numpy(), assets_to_trade)

def test(Data,
         daily_dates,
         model,
         preprocess,
         config,
         residual_weights=None,
         log_dev_progress_freq=50, log_plot_freq=199,
         num_epochs=100, lr=0.001, batchsize=150,
         early_stopping=False,
         save_params=True,
         device='cuda',
         output_path=os.path.join(os.getcwd(), 'results', 'Unknown'), model_tag='Unknown',
         lookback=30, retrain_freq=250, length_training=1000, rolling_retrain=True,
         parallelize=True,
         device_ids=[0,1,2,3,4,5,6,7],
         trans_cost=0, hold_cost=0,
         force_retrain=False,
         objective="sharpe"):

    # choose assets with at least `lookback` non-missing obs
    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback
    logging.info(f"test(): assets_to_trade.shape {assets_to_trade.shape}")
    Data = Data[:, assets_to_trade]
    T, N = Data.shape
    returns = np.zeros(T - length_training)
    turnovers = np.zeros(T - length_training)
    short_proportions = np.zeros(T - length_training)
    all_weights = np.zeros((T - length_training, len(assets_to_trade)))

    # rolling subperiods — start only after lookback days
    start_idx = max(lookback, 0)
    num_subperiods = int((T - length_training - start_idx) / retrain_freq) + 1

    for t in range(num_subperiods):
        logging.info(f'AT SUBPERIOD {t}/{num_subperiods}')
        offset = start_idx + t * retrain_freq

        data_train_t = Data[offset:length_training + offset]
        data_test_t = Data[length_training + offset - lookback:
                           min(length_training + (t + 1) * retrain_freq + start_idx, T)]
        residual_weights_train_t = None if residual_weights is None else residual_weights[offset:length_training + offset]
        residual_weights_test_t = None if residual_weights is None else residual_weights[length_training + offset - lookback:
                                                                                        min(length_training + (t + 1) * retrain_freq + start_idx, T)]
        model_tag_t = model_tag + f'__subperiod{t}'

        if rolling_retrain or t == 0:
            model_t = model(logdir=output_path, **config['model'])
            (rets_t, turns_t, shorts_t, weights_t, a2t) = train(
                model_t,
                preprocess=preprocess,
                data_train=data_train_t,
                data_dev=data_test_t,
                residual_weights_train=residual_weights_train_t,
                residual_weights_dev=residual_weights_test_t,
                log_dev_progress_freq=log_dev_progress_freq,
                num_epochs=num_epochs,
                force_retrain=force_retrain,
                optimizer_name=config['optimizer_name'],
                optimizer_opts=config['optimizer_opts'],
                early_stopping=early_stopping,
                save_params=save_params,
                output_path=output_path,
                model_tag=model_tag_t,
                device=device,
                lookback=lookback,
                log_plot_freq=log_plot_freq,
                parallelize=parallelize,
                device_ids=device_ids,
                batchsize=batchsize,
                trans_cost=trans_cost,
                hold_cost=hold_cost,
                objective=objective,
            )
        else:
            (rets_t, _, _, turns_t, shorts_t, weights_t, a2t) = get_returns(
                model_t,
                preprocess=preprocess,
                objective=objective,
                data_test=data_test_t,
                residual_weights=residual_weights_test_t,
                device=device,
                lookback=lookback,
                trans_cost=trans_cost,
                hold_cost=hold_cost,
            )

        # ✅ safe slice assignment
        end_idx = min(offset - start_idx + retrain_freq, T - length_training)
        slice_len = end_idx - (offset - start_idx)

        returns[offset - start_idx:end_idx] = rets_t[:slice_len]
        turnovers[offset - start_idx:end_idx] = turns_t[:slice_len]
        short_proportions[offset - start_idx:end_idx] = shorts_t[:slice_len]

        # expand weights to full asset set for saving
        w = np.zeros((slice_len, len(a2t)))
        rows_to_copy = min(slice_len, weights_t.shape[0])
        w[:rows_to_copy, a2t] = weights_t[:rows_to_copy]

        if rows_to_copy < slice_len:
            logging.warning(
                f"[test] weights_t shorter ({weights_t.shape[0]}) than slice_len ({slice_len}), "
                f"padding remaining {slice_len - rows_to_copy} rows with zeros "
                f"(subperiod={t}, offset={offset}, T={T}, N={N})"
            )

        all_weights[offset - start_idx:end_idx, assets_to_trade] = w

        if 'cpu' not in device:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    logging.info('TRAIN/TEST COMPLETE')
    cumRets = np.cumprod(1 + returns)
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], cumRets, marker='None', linestyle='solid')
    plt.savefig(os.path.join(output_path, model_tag + "_cumulative-returns.png"))

    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], turnovers, marker='None', linestyle='solid')
    plt.savefig(os.path.join(output_path, model_tag + "_turnover.png"))

    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], short_proportions, marker='None', linestyle='solid')
    plt.savefig(os.path.join(output_path, model_tag + "_short-proportion.png"))

    np.save(os.path.join(output_path, 'WeightsComplete_' + model_tag + '.npy'), all_weights)

    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = 0.0 if full_std <= 0 else full_ret / full_std
    logging.info(f"==> Sharpe: {full_sharpe*np.sqrt(252):.2f}, "
                 f"ret: {full_ret*252:.4f}, "
                 f"std: {full_std*np.sqrt(252):.4f}, "
                 f"turnover: {np.mean(turnovers):.4f}, "
                 f"short_proportion: {np.mean(short_proportions):.4f}")

    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions


def estimate(Data,
             daily_dates,
             model,
             preprocess,
             config,
             residual_weights=None,
             log_dev_progress_freq=50, log_plot_freq=199,
             num_epochs=100, lr=0.001, batchsize=150,
             early_stopping=False,
             save_params=True,
             device='cuda',
             output_path=os.path.join(os.getcwd(), 'results', 'Unknown'), model_tag='Unknown',
             lookback=30, length_training=1000, test_size=125,
             parallelize=True,
             device_ids=[0,1,2,3,4,5,6,7],
             trans_cost=0, hold_cost=0,
             force_retrain=True,
             objective="sharpe",
             estimate_start_idx=0):

    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback
    Data = Data[:, assets_to_trade]
    T, N = Data.shape
    returns = np.zeros(length_training)
    turnovers = np.zeros(length_training)
    short_proportions = np.zeros(length_training)
    all_weights = np.zeros((length_training, len(assets_to_trade)))

    logging.info(f"ESTIMATING {estimate_start_idx}:{min(estimate_start_idx+length_training,T)}")
    logging.info(f"TESTING {estimate_start_idx+length_training-lookback}:{min(estimate_start_idx+length_training+test_size,T)}")
    data_train = Data[estimate_start_idx:min(estimate_start_idx + length_training, T)]
    data_dev = Data[estimate_start_idx + length_training - lookback:min(estimate_start_idx + length_training + test_size, T)]
    residual_weights_train = None if residual_weights is None else residual_weights[estimate_start_idx:min(estimate_start_idx + length_training, T)]
    residual_weights_dev = None if residual_weights is None else residual_weights[estimate_start_idx + length_training - lookback:min(estimate_start_idx + length_training + test_size, T)]
    del residual_weights
    del Data
    model_tag = model_tag + f'__estimation{estimate_start_idx}-{length_training}-{test_size}'

    model1 = model(logdir=output_path, **config['model'])
    rets, turns, shorts, weights = train(
        model1,
        preprocess=preprocess,
        data_train=data_train,
        data_dev=data_dev,
        residual_weights_train=residual_weights_train,
        residual_weights_dev=residual_weights_dev,
        log_dev_progress_freq=log_dev_progress_freq,
        num_epochs=num_epochs,
        force_retrain=force_retrain,
        optimizer_name=config['optimizer_name'],
        optimizer_opts=config['optimizer_opts'],
        early_stopping=early_stopping,
        save_params=save_params,
        output_path=output_path,
        model_tag=model_tag,
        device=device,
        lookback=lookback,
        log_plot_freq=log_plot_freq,
        parallelize=parallelize,
        device_ids=device_ids,
        batchsize=batchsize,
        trans_cost=trans_cost,
        hold_cost=hold_cost,
        objective=objective,
    )

    returns = rets
    turnovers = turns
    short_proportions = shorts
    all_weights = weights
    if 'cpu' not in device:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    logging.info('ESTIMATION COMPLETE')

    np.save(os.path.join(output_path, 'WeightsComplete_' + model_tag + '.npy'), all_weights)

    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = 0.0 if full_std <= 0 else full_ret / full_std
    logging.info(f"==> Sharpe: {full_sharpe*np.sqrt(252):.2f}, "
                 f"ret: {full_ret*252:.4f}, "
                 f"std: {full_std*np.sqrt(252):.4f}, "
                 f"turnover: {np.mean(turnovers):.4f}, "
                 f"short_proportion: {np.mean(short_proportions):.4f}")

    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions
