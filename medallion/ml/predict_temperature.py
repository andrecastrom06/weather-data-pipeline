from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import sys
import uuid

from sklearn.linear_model import LinearRegression
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medallion.utils.connection import engine


@dataclass
class EvalResult:
    run_id: str
    executed_at_utc: datetime
    step_minutes: int
    horizon_minutes: int
    model_name: str
    mae: float
    rmse: float
    n_train: int
    n_test: int


@dataclass
class RunSummary:
    run_id: str
    executed_at_utc: datetime
    status: str
    message: str
    step_minutes: int | None
    silver_rows: int
    results_count: int
    forecasts_count: int


def load_silver_data() -> pd.DataFrame:
    query = """
        SELECT
            datetime,
            temperature,
            humidity,
            pressure,
            wind_speed
        FROM silver_olinda_weather
        ORDER BY datetime
    """
    df = pd.read_sql(query, con=engine)

    if df.empty:
        raise ValueError("Tabela silver_olinda_weather vazia. Rode o pipeline ETL antes.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "temperature", "humidity", "pressure", "wind_speed"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def infer_step_minutes(df: pd.DataFrame) -> int:
    diffs = df["datetime"].diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]

    if diffs.empty:
        return 5

    step = int(round(float(diffs.median())))
    return max(step, 1)


def make_features(
    df: pd.DataFrame,
    horizon_minutes: int,
    step_minutes: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if horizon_minutes < step_minutes:
        raise ValueError(f"Horizonte {horizon_minutes} < passo da serie ({step_minutes} min).")

    horizon_steps = int(round(horizon_minutes / step_minutes))
    lag_steps = [1, 2, 3, 6, 12, 24]

    work = df.copy()
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        for lag in lag_steps:
            work[f"{col}_lag_{lag}"] = work[col].shift(lag)

    work["hour"] = work["datetime"].dt.hour
    work["day_of_week"] = work["datetime"].dt.dayofweek
    work["observed_at"] = work["datetime"]
    work["forecast_for"] = work["datetime"].shift(-horizon_steps)
    work["target_temperature"] = work["temperature"].shift(-horizon_steps)
    work = work.dropna().reset_index(drop=True)

    meta = work[["observed_at", "forecast_for"]].copy()
    y = work["target_temperature"].astype(float)
    drop_cols = ["datetime", "observed_at", "forecast_for", "target_temperature"]
    X = work.drop(columns=drop_cols)

    if X.empty:
        raise ValueError("Sem dados suficientes para gerar features. Aumente historico na silver.")

    return X, y, meta


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    cut = int(len(X) * (1 - test_ratio))
    if cut <= 0 or cut >= len(X):
        raise ValueError("Split temporal invalido. Verifique volume de dados.")

    X_train = X.iloc[:cut]
    X_test = X.iloc[cut:]
    y_train = y.iloc[:cut]
    y_test = y.iloc[cut:]
    meta_train = meta.iloc[:cut]
    meta_test = meta.iloc[cut:]
    return X_train, X_test, y_train, y_test, meta_train, meta_test


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def available_models() -> dict[str, object]:
    models: dict[str, object] = {}

    try:
        models["LinearRegression"] = LinearRegression()
    except Exception:
        pass

    try:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
    except Exception:
        pass

    if not models:
        raise ImportError("Nenhum modelo disponivel")

    return models


def run_forecast() -> tuple[RunSummary, list[EvalResult], pd.DataFrame, pd.DataFrame]:
    run_id = str(uuid.uuid4())
    executed_at_utc = datetime.now(timezone.utc)

    attempt_rows: list[dict] = []
    results: list[EvalResult] = []
    forecast_frames: list[pd.DataFrame] = []

    try:
        df = load_silver_data()
    except Exception as exc:
        summary = RunSummary(
            run_id=run_id,
            executed_at_utc=executed_at_utc,
            status="no_data",
            message=f"Falha ao carregar silver: {exc}",
            step_minutes=None,
            silver_rows=0,
            results_count=0,
            forecasts_count=0,
        )
        attempt_rows.append(
            {
                "run_id": run_id,
                "executed_at_utc": executed_at_utc,
                "horizon_minutes": None,
                "model_name": None,
                "status": "failed",
                "error_message": str(exc),
                "n_train": None,
                "n_test": None,
                "mae": None,
                "rmse": None,
            }
        )
        return summary, results, pd.DataFrame(), pd.DataFrame(attempt_rows)

    step_minutes = infer_step_minutes(df)
    print(f"Passo temporal inferido: {step_minutes} minuto(s)")
    print(f"Linhas disponiveis para treino: {len(df)}")

    horizons = [30, 60, 120, 180]

    try:
        models = available_models()
    except Exception as exc:
        summary = RunSummary(
            run_id=run_id,
            executed_at_utc=executed_at_utc,
            status="models_unavailable",
            message=f"Modelos indisponiveis: {exc}",
            step_minutes=step_minutes,
            silver_rows=len(df),
            results_count=0,
            forecasts_count=0,
        )
        attempt_rows.append(
            {
                "run_id": run_id,
                "executed_at_utc": executed_at_utc,
                "horizon_minutes": None,
                "model_name": None,
                "status": "failed",
                "error_message": str(exc),
                "n_train": None,
                "n_test": None,
                "mae": None,
                "rmse": None,
            }
        )
        return summary, results, pd.DataFrame(), pd.DataFrame(attempt_rows)

    for horizon in horizons:
        try:
            X, y, meta = make_features(df, horizon_minutes=horizon, step_minutes=step_minutes)
            X_train, X_test, y_train, y_test, _, meta_test = time_split(X, y, meta, test_ratio=0.2)
        except Exception as exc:
            print(f"[h={horizon}min] pulado: {exc}")
            attempt_rows.append(
                {
                    "run_id": run_id,
                    "executed_at_utc": executed_at_utc,
                    "horizon_minutes": horizon,
                    "model_name": None,
                    "status": "skipped",
                    "error_message": str(exc),
                    "n_train": None,
                    "n_test": None,
                    "mae": None,
                    "rmse": None,
                }
            )
            continue

        print(f"\n[h={horizon}min] train={len(X_train)} test={len(X_test)}")

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = np.asarray(model.predict(X_test), dtype=float)
                y_true = y_test.to_numpy(dtype=float)

                result = EvalResult(
                    run_id=run_id,
                    executed_at_utc=executed_at_utc,
                    step_minutes=step_minutes,
                    horizon_minutes=horizon,
                    model_name=name,
                    mae=mae(y_true, pred),
                    rmse=rmse(y_true, pred),
                    n_train=len(X_train),
                    n_test=len(X_test),
                )
                results.append(result)
                print(f"  {name:<16} MAE={result.mae:.4f} RMSE={result.rmse:.4f}")

                attempt_rows.append(
                    {
                        "run_id": run_id,
                        "executed_at_utc": executed_at_utc,
                        "horizon_minutes": horizon,
                        "model_name": name,
                        "status": "success",
                        "error_message": None,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "mae": result.mae,
                        "rmse": result.rmse,
                    }
                )

                model_forecasts = pd.DataFrame(
                    {
                        "run_id": run_id,
                        "executed_at_utc": executed_at_utc,
                        "step_minutes": step_minutes,
                        "horizon_minutes": horizon,
                        "model_name": name,
                        "observed_at": pd.to_datetime(meta_test["observed_at"], errors="coerce"),
                        "forecast_for": pd.to_datetime(meta_test["forecast_for"], errors="coerce"),
                        "temperature_actual": y_true,
                        "temperature_predicted": pred,
                    }
                )
                model_forecasts["error"] = (
                    model_forecasts["temperature_predicted"] - model_forecasts["temperature_actual"]
                )
                model_forecasts["absolute_error"] = model_forecasts["error"].abs()
                forecast_frames.append(model_forecasts)
            except Exception as exc:
                print(f"  {name:<16} falhou: {exc}")
                attempt_rows.append(
                    {
                        "run_id": run_id,
                        "executed_at_utc": executed_at_utc,
                        "horizon_minutes": horizon,
                        "model_name": name,
                        "status": "failed",
                        "error_message": str(exc),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "mae": None,
                        "rmse": None,
                    }
                )

    if forecast_frames:
        forecast_df = pd.concat(forecast_frames, ignore_index=True)
    else:
        forecast_df = pd.DataFrame()

    attempts_df = pd.DataFrame(attempt_rows)
    if results:
        status = "ok"
        message = "Execucao concluida com resultados de ML."
    else:
        status = "partial_no_predictions"
        message = "Execucao concluida sem previsoes validas; tentativas registradas."

    summary = RunSummary(
        run_id=run_id,
        executed_at_utc=executed_at_utc,
        status=status,
        message=message,
        step_minutes=step_minutes,
        silver_rows=len(df),
        results_count=len(results),
        forecasts_count=len(forecast_df),
    )

    return summary, results, forecast_df, attempts_df


def print_best_by_horizon(results: list[EvalResult]) -> None:
    if not results:
        print("Sem resultados para comparar.")
        return

    print("\nMelhor modelo por horizonte (menor RMSE):")
    df = pd.DataFrame([r.__dict__ for r in results])

    for horizon in sorted(df["horizon_minutes"].unique()):
        row = (
            df[df["horizon_minutes"] == horizon]
            .sort_values("rmse", ascending=True)
            .iloc[0]
        )
        print(
            f"  {int(row['horizon_minutes'])}min -> {row['model_name']} "
            f"(RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f})"
        )


def save_ml_outputs(
    summary: RunSummary,
    results: list[EvalResult],
    forecast_df: pd.DataFrame,
    attempts_df: pd.DataFrame,
) -> None:
    run_df = pd.DataFrame([summary.__dict__])
    run_df.to_sql(
        name="ml_temperature_runs",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    if not attempts_df.empty:
        attempts_df.to_sql(
            name="ml_temperature_attempts",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
        )

    if not results:
        print("Execucao sem resultados de modelos; status e tentativas foram salvos no banco.")
        return

    metrics_df = pd.DataFrame([r.__dict__ for r in results])
    best_models = (
        metrics_df.sort_values(["horizon_minutes", "rmse"], ascending=[True, True])
        .groupby("horizon_minutes", as_index=False)
        .first()[["horizon_minutes", "model_name"]]
        .rename(columns={"model_name": "best_model_name"})
    )

    metrics_df = metrics_df.merge(best_models, on="horizon_minutes", how="left")
    metrics_df["is_best_model_horizon"] = metrics_df["model_name"] == metrics_df["best_model_name"]
    metrics_df = metrics_df.drop(columns=["best_model_name"])

    metrics_df.to_sql(
        name="ml_temperature_model_metrics",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    if forecast_df.empty:
        print("Sem previsoes para salvar; metricas, status e tentativas foram salvos.")
        return

    forecast_df = forecast_df.merge(best_models, on="horizon_minutes", how="left")
    forecast_df["is_best_model_horizon"] = forecast_df["model_name"] == forecast_df["best_model_name"]
    forecast_df = forecast_df.drop(columns=["best_model_name"])

    forecast_df.to_sql(
        name="ml_temperature_forecasts",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    print(
        "Saida de ML salva no banco: ml_temperature_runs, ml_temperature_attempts, "
        "ml_temperature_model_metrics e ml_temperature_forecasts."
    )


def main():
    try:
        summary, all_results, forecast_df, attempts_df = run_forecast()
    except Exception as exc:
        fallback_summary = RunSummary(
            run_id=str(uuid.uuid4()),
            executed_at_utc=datetime.now(timezone.utc),
            status="fatal_error",
            message=str(exc),
            step_minutes=None,
            silver_rows=0,
            results_count=0,
            forecasts_count=0,
        )
        fallback_attempts = pd.DataFrame(
            [
                {
                    "run_id": fallback_summary.run_id,
                    "executed_at_utc": fallback_summary.executed_at_utc,
                    "horizon_minutes": None,
                    "model_name": None,
                    "status": "failed",
                    "error_message": str(exc),
                    "n_train": None,
                    "n_test": None,
                    "mae": None,
                    "rmse": None,
                }
            ]
        )
        save_ml_outputs(fallback_summary, [], pd.DataFrame(), fallback_attempts)
        raise

    print_best_by_horizon(all_results)
    save_ml_outputs(summary, all_results, forecast_df, attempts_df)


if __name__ == "__main__":
    main()
