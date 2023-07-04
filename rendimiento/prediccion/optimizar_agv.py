import numpy as np
import optuna
import torch
import random

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
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.tcn_model import TCNModel

from darts.utils.likelihood_models import GaussianLikelihood
from darts.metrics import mae

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
        ' |> aggregateWindow(every: 200ms, fn: last, createEmpty: false)' \
        ' |> filter(fn: (r) => r._measurement == "test" and r.type == "value")' \
        ' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

    df = query_api.query_data_frame(query=query)
    df.drop(columns=['result', 'table', '_start', '_stop'])

    df['_time'] = df['_time'].values.astype('<M8[us]')
    df.head()
    return df

def get_series(tag: str, df: DataFrame) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, "_time", tag, freq="200ms").astype(np.float32)
    series = fill_missing_values(series=series)

    return series

def objective_transformer(trial):
    in_len = trial.suggest_int("in_len_trans", 51, 200)
    # out_len = trial.suggest_categorical("out_len", [30, 50, 100, 150])
    out_len = 50
    d_model = trial.suggest_categorical("d_model_trans", [16, 34, 64, 128])
    nhead = trial.suggest_categorical("nhead_trans", [1, 2, 4, 8, 16])
    enc_layers = trial.suggest_int("enc_layers_trans", 1, 20)
    dec_layers = trial.suggest_int("dec_layers_trans", 1, 20)
    dim_ffwd = trial.suggest_categorical("dim_ffwd_trans", [128, 256, 512, 1024])
    dropout = trial.suggest_float("dropout_trans", 0.01, 0.5)
    activation = trial.suggest_categorical("activation_trans", ["GLU", "ReLU", "Bilinear", "SwiGLU"])
    norm_type = trial.suggest_categorical("norm_type_trans", [None, "LayerNorm", "RMSNorm", "LayerNormNoBias"])

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=10, verbose=True)
    callbacks = [pruner, early_stopper]

    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
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

    covariates = train_sr_scaled.stack(train_sl_scaled).stack(train_ei_scaled)
    val_covariates = series_sr_scaled.stack(series_ei_scaled).stack(series_sl_scaled)

    try:
        model.fit(
            series=train_ed_scaled,
            val_series=val_ed_scaled,
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
        maes = mae(val_ed_scaled, preds, n_jobs=-1, verbose=True)
        mae_val = np.mean(maes)

        print(mae_val)
        return mae_val if mae_val != np.nan else float("inf")
    except Exception as e:
        print(e)
        return float("inf")

def objective_nhits(trial):
    in_len = trial.suggest_int("in_len_nhits", 51, 200)
    # out_len = trial.suggest_categorical("out_len", [30, 50, 100, 150])
    out_len = 50
    num_stacks = trial.suggest_int("num_stacks_nhits", 1, 15)
    num_blocks = trial.suggest_int("num_blocks_nhits", 1, 15)
    num_layers = trial.suggest_int("num_layers_nhits", 1, 15)
    layer_widths = trial.suggest_int("layer_widths_nhits", 1, 15)
    dropout = trial.suggest_float("dropout_nhits", 0.01, 0.5)
    activation = trial.suggest_categorical("activation_nhits", ["ReLU", "RReLU", "PReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid"])
    max_pool_1d = trial.suggest_categorical("max_pool_1d_nhits", [True, False])

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=10, verbose=True)
    callbacks = [pruner, early_stopper]

    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    model = NHiTSModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        MaxPool1d=max_pool_1d,
        dropout=dropout,
        activation=activation,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="nhits_model",
        force_reset=True,
        save_checkpoints=True,
    )

    covariates = train_sr_scaled.stack(train_sl_scaled).stack(train_ei_scaled)
    val_covariates = series_sr_scaled.stack(series_ei_scaled).stack(series_sl_scaled)

    try:
        model.fit(
            series=train_ed_scaled,
            val_series=val_ed_scaled,
            past_covariates=covariates,
            val_past_covariates=val_covariates,
            num_loader_workers=num_workers
        )

        model = NHiTSModel.load_from_checkpoint("nhits_model")

        preds = model.predict(
            series=train_ed_scaled,
            past_covariates=covariates,
            n=out_len
        )
        maes = mae(val_ed_scaled, preds, n_jobs=-1, verbose=True)
        mae_val = np.mean(maes)

        print(mae_val)
        return mae_val if mae_val != np.nan else float("inf")
    except Exception as e:
        print(e)
        return float("inf")
    
def objective_tcn(trial):
    in_len = trial.suggest_int("in_len_tcn", 51, 200)
    # out_len = trial.suggest_categorical("out_len", [30, 50, 100, 150])
    out_len = 50
    kernel_size = trial.suggest_int("kernel_size_tcn", 1, 10)
    num_filters = trial.suggest_int("num_filters_tcn", 1, 10)
    weight_norm = trial.suggest_categorical("weight_norm_tcn", [True, False])
    dilation_base = trial.suggest_int("dilation_base_tcn", 1, 10)
    num_layers = trial.suggest_int("num_layers_tcn", 1, 10)
    dropout = trial.suggest_float("dropout_tcn", 0.01, 0.5)

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=10, verbose=True)
    callbacks = [pruner, early_stopper]

    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=50,
        kernel_size=kernel_size,
        num_filters=num_filters,
        num_layers=num_layers,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )

    covariates = train_sr_scaled.stack(train_sl_scaled).stack(train_ei_scaled)
    val_covariates = series_sr_scaled.stack(series_ei_scaled).stack(series_sl_scaled)

    try:
        model.fit(
            series=train_ed_scaled,
            val_series=val_ed_scaled,
            past_covariates=covariates,
            val_past_covariates=val_covariates,
            num_loader_workers=num_workers
        )

        model = NHiTSModel.load_from_checkpoint("tcn_model")

        preds = model.predict(
            series=train_ed_scaled,
            past_covariates=covariates,
            n=out_len
        )
        maes = mae(val_ed_scaled, preds, n_jobs=-1, verbose=True)
        mae_val = np.mean(maes)

        print(mae_val)
        return mae_val if mae_val != np.nan else float("inf")
    except Exception as e:
        print(e)
        return float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

df = query_dataframe()

series_ed = get_series('encoder_derecho', df)
series_ed = fill_missing_values(Diff(lags=5, dropna=False).fit_transform(series=series_ed))
series_ei = get_series('encoder_izquierdo', df)
series_ei = fill_missing_values(Diff(lags=5, dropna=False).fit_transform(series=series_ei))
series_sr = get_series('out.set_speed_right', df)
series_sl = get_series('out.set_speed_left', df)

train_ed, val_ed = series_ed[:-VAL_SIZE], series_ed[-VAL_SIZE:]
train_ei, val_ei = series_ei[:-VAL_SIZE], series_ei[-VAL_SIZE:]
train_sr, val_sr = series_sr[:-VAL_SIZE], series_sr[-VAL_SIZE:]
train_sl, val_sl = series_sl[:-VAL_SIZE], series_sl[-VAL_SIZE:]

scaler_ed, scaler_ei, scaler_sr, scaler_sl = Scaler(), Scaler(), Scaler(), Scaler()
train_ed_scaled = scaler_ed.fit_transform(train_ed)
val_ed_scaled = scaler_ed.transform(val_ed)
series_ed_scaled = scaler_ed.transform(series_ed)
train_ei_scaled = scaler_ei.fit_transform(train_ei)
val_ei_scaled = scaler_ei.transform(val_ei)
series_ei_scaled = scaler_ei.transform(series_ei)
train_sr_scaled = scaler_sr.fit_transform(train_sr)
val_sr_scaled = scaler_sr.transform(val_sr)
series_sr_scaled = scaler_sr.transform(series_sr)
train_sl_scaled = scaler_sl.fit_transform(train_sl)
val_sl_scaled = scaler_sl.transform(val_sl)
series_sl_scaled = scaler_sl.transform(series_sl)

f = open("optimizacion_logs.txt", "w")
study = optuna.create_study(direction="minimize")
study.optimize(objective_transformer, n_trials=100, timeout=7200, callbacks=[print_callback])
f.write("Transformer\n")
f.write(f"Best value: {study.best_value}, Best params: {study.best_trial.params}\n")

study = optuna.create_study(direction="minimize")
study.optimize(objective_nhits, n_trials=100, timeout=7200, callbacks=[print_callback])
f.write("N-HiTS\n")
f.write(f"Best value: {study.best_value}, Best params: {study.best_trial.params}\n")

study = optuna.create_study(direction="minimize")
study.optimize(objective_tcn, n_trials=100, timeout=7200, callbacks=[print_callback])
f.write("TCN\n")
f.write(f"Best value: {study.best_value}, Best params: {study.best_trial.params}\n")

f.close()