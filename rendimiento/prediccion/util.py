import time

from matplotlib import pyplot as plt

from darts.metrics.metrics import mae, mase, dtw_metric

def test_univariate(model, nombre, val_size, salida, series, train_series, val_series):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(train_series)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        num_predictions = int(val_size / salida)
        start_pred = time.time()
        preds = model.predict(series=train_series, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=train_series) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, n=salida)
            preds = preds.append(pred)

            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=new_series) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces

def test_univariate_ml(model, nombre, val_size, salida, series, train_series, val_series):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(train_series, epochs=200)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        num_predictions = int(val_size / salida)
        start_pred = time.time()
        preds = model.predict(series=train_series, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=train_series) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, n=salida)
            preds = preds.append(pred)

            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=new_series) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces

def test_covariante(model, nombre, val_size, salida, series, train_series, covariates_series, train_covariates, val_series, val_covariates):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(series=train_series, past_covariates=train_covariates, epochs=200)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        num_predictions = int(val_size / salida)
        start_pred = time.time()
        preds = model.predict(series=train_series, past_covariates=train_covariates, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=train_series) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            new_covariates = covariates_series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, past_covariates=new_covariates, n=salida)
            preds = preds.append(pred)

            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=new_series) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces

def test_multivariate(model, nombre, val_size, salida, series, train_multi_series, train_series, val_multi_series, val_series):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(series=train_multi_series, epochs=200)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        num_predictions = int(val_size / salida)
        start_pred = time.time()
        preds = model.predict(series=train_multi_series, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred
        preds = preds["encoder_derecho"]

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=train_series) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, n=salida)
            pred = pred["encoder_derecho"]
            preds = preds.append(pred)


            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=new_series["encoder_derecho"]) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces

def test_univariate_ml_scaled(model, nombre, val_size, salida, series, train_series, val_series, val_series_scaled, scaler):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(train_series, epochs=200)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        num_predictions = int(val_size / salida)
        start_pred = time.time()
        preds = model.predict(series=train_series, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred
        preds = scaler.inverse_transform(preds)

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=scaler.inverse_transform(train_series)) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, n=salida)
            pred = scaler.inverse_transform(pred)
            preds = preds.append(pred)

            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=scaler.inverse_transform(new_series)) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces

def test_covariante_scaled(model, nombre, val_size, salida, series, train_series, covariates_series, train_covariates, val_series, val_series_scaled, val_covariates, scaler):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(series=train_series, past_covariates=train_covariates, epochs=200)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        num_predictions = int(val_size / salida)
        start_pred = time.time()
        preds = model.predict(series=train_series, past_covariates=train_covariates, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred
        preds = scaler.inverse_transform(preds)

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=scaler.inverse_transform(train_series)) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            new_covariates = covariates_series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, past_covariates=new_covariates, n=salida)
            pred = scaler.inverse_transform(pred)
            preds = preds.append(pred)

            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=scaler.inverse_transform(new_series)) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces

def test_multivariate_scaled(model, nombre, val_size, salida, series, train_multi_series, train_series, val_multi_series, val_series, scaler_multi, scaler):
    m_mae = 0
    m_mase = 0
    m_dtw = 0
    t_entrenamiento = 0
    t_prediccion = 0

    veces = 5

    for i in range(0, veces):
        start_fit = time.time()
        model.fit(series=train_multi_series, epochs=200)
        end_fit = time.time()
        t_entrenamiento += end_fit - start_fit

        start_pred = time.time()
        num_predictions = int(val_size / salida)
        preds = model.predict(series=train_multi_series, n=salida)
        end_pred = time.time()
        t_prediccion += end_pred - start_pred
        preds = scaler_multi.inverse_transform(preds)
        preds = preds["encoder_derecho"]

        m_mae += mae(actual_series=val_series[:salida], pred_series=preds) / num_predictions
        m_mase += mase(actual_series=val_series[:salida], pred_series=preds, insample=train_series) / num_predictions
        m_dtw += dtw_metric(actual_series=val_series[:salida], pred_series=preds) / num_predictions

        for i in range(1, num_predictions):
            new_series = series[:-(val_size - i * salida)]
            pred = model.predict(series=new_series, n=salida)
            pred = scaler_multi.inverse_transform(pred)
            pred = pred["encoder_derecho"]
            preds = preds.append(pred)

            m_mae += mae(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions
            m_mase += mase(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred, insample=scaler_multi.inverse_transform(new_series)["encoder_derecho"]) / num_predictions
            m_dtw += dtw_metric(actual_series=val_series[i * salida:(i + 1) * salida], pred_series=pred) / num_predictions

    preds.plot()
    plt.legend(["Entrenamiento", "Validación", "Predicción"], fontsize=15)
    plt.xlabel("Tiempo", fontsize=20)
    plt.ylabel("Valor", fontsize=20)
    plt.savefig(nombre + str(salida) + ".png")
    plt.clf()

    return m_mae / veces, m_mase / veces, m_dtw / veces, t_entrenamiento / veces, t_prediccion / veces