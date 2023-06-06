import numpy as np
import optuna
import torch
import logging

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from influxdb_client import InfluxDBClient

from pandas import DataFrame
from darts.dataprocessing.transformers import Diff
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.transformer_model import TransformerModel

from darts.utils.likelihood_models import GaussianLikelihood
from darts.metrics import smape

import matplotlib.pyplot as plt
import torch

VAL_SIZE = 700

torch.set_float32_matmul_precision('high')

token = "u_nT6lvnTJEfY1xrcGF7E6ypuHKXDLoGOKXm580Q2pyFwNYv8CY_yFGUkCgjPep387EWuhE3p90EQaYFkW5Zww=="
bucket = "AGV"
org = "TFG"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()

def query_dataframe() -> DataFrame:
    query = 'from(bucket:"AGV")' \
        ' |> range(start:2023-05-30, stop:2023-05-31)' \
        ' |> aggregateWindow(every: 100ms, fn: last, createEmpty: false)' \
        ' |> filter(fn: (r) => r._measurement == "test" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

    df = query_api.query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])

    df['_time'] = df['_time'].values.astype('<M8[us]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, "_time", tag, freq="100ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series

def objective(trial):
    in_len = trial.suggest_categorical("in_len", [50, 100, 200, 300])
    # out_len = trial.suggest_categorical("out_len", [30, 50, 100, 150])
    out_len = VAL_SIZE
    d_model = trial.suggest_categorical("d_model", [16, 34, 64, 128])
    nhead = trial.suggest_categorical("nhead", [1, 2, 4, 8, 16])
    enc_layers = trial.suggest_int("enc_layers", 1, 10)
    dec_layers = trial.suggest_int("dec_layers", 1, 10)
    dim_ffwd = trial.suggest_categorical("dim_ffwd", [128, 256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.01, 0.7)
    activation = trial.suggest_categorical("activation", ["GLU", "ReLU", "Bilinear", "SwiGLU"])
    norm_type = trial.suggest_categorical("norm_type", [None, "LayerNorm", "RMSNorm", "LayerNormNoBias"])

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=10, verbose=True)
    callbacks = [pruner, early_stopper]

    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            # "gpus": -1,
            # "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    model = TransformerModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_ffwd,
        dropout=dropout,
        activation=activation,
        likelihood=GaussianLikelihood(),
        norm_type=norm_type,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="transformer_model",
        force_reset=True,
        save_checkpoints=True
    )

    model_val_set_ed = scaler_ed.transform(series_ed[-(VAL_SIZE + in_len) :])
    model_val_set_ei = scaler_ed.transform(series_ei[-(VAL_SIZE + in_len) :])
    model_val_set_sr = scaler_ed.transform(series_sr[-(VAL_SIZE + in_len) :])
    model_val_set_sl = scaler_ed.transform(series_sl[-(VAL_SIZE + in_len) :])
    # val = model_val_set_ed.stack(model_val_set_ei).stack(model_val_set_sr).stack(model_val_set_sl)
    # train = train_ed_scaled.stack(train_ei_scaled).stack(train_sr_scaled).stack(train_sl_scaled)
    covariates = train_sr_scaled.stack(train_sl_scaled).stack(train_ei_scaled)
    val_covariates = model_val_set_sr.stack(model_val_set_sl).stack(model_val_set_ei)

    try:
        model.fit(
            series=train_ed_scaled,
            val_series=model_val_set_ed,
            past_covariates=covariates,
            val_past_covariates=val_covariates,
            num_loader_workers=num_workers
        )

        model = TransformerModel.load_from_checkpoint("transformer_model")

        preds = model.predict(
            series=train_ed_scaled,
            past_covariates=covariates,
            n=out_len
        )
        smapes = smape(val_ed_scaled, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        return smape_val if smape_val != np.nan else float("inf")
    except Exception as e:
        logging.info(e)
        return float("inf")

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

df = query_dataframe()

series_ed = get_series('encoder_derecho', df)
series_ed = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ed))
series_ei = get_series('encoder_izquierdo', df)
series_ei = fill_missing_values(Diff(lags=1, dropna=False).fit_transform(series=series_ei))
series_sr = get_series('out.set_speed_right', df)
series_sl = get_series('out.set_speed_left', df)

scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(MaxAbsScaler()), Scaler(MaxAbsScaler()), Scaler(MaxAbsScaler()), Scaler(MaxAbsScaler())

train_ed, val_ed = series_ed[:-VAL_SIZE], series_ed[-VAL_SIZE:]
train_ei, val_ei = series_ei[:-VAL_SIZE], series_ei[-VAL_SIZE:]
train_sr, val_sr = series_sr[:-VAL_SIZE], series_sr[-VAL_SIZE:]
train_sl, val_sl = series_sl[:-VAL_SIZE], series_sl[-VAL_SIZE:]

train_ed_scaled = scaler_ed.fit_transform(train_ed)
val_ed_scaled = scaler_ed.transform(val_ed)
train_ei_scaled = scaler_ei.fit_transform(train_ei)
val_ei_scaled = scaler_ei.transform(val_ei)
train_sr_scaled = scaler_sr.fit_transform(train_sr)
val_sr_scaled = scaler_sr.transform(val_sr)
train_sl_scaled = scaler_sl.fit_transform(train_sl)
val_sl_scaled = scaler_sl.transform(val_sl)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, callbacks=[print_callback])