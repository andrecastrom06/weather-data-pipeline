from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
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
    horizon_minutes: int
    model_name: str
    mae: float
    rmse: float
    n_train: int
    n_test: int


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


def make_features(df: pd.DataFrame, horizon_minutes: int, step_minutes: int) -> tuple[pd.DataFrame, pd.Series]:
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

    work["target_temperature"] = work["temperature"].shift(-horizon_steps)
    work = work.dropna().reset_index(drop=True)

    y = work["target_temperature"]
    drop_cols = ["datetime", "target_temperature"]
    X = work.drop(columns=drop_cols)

    if X.empty:
        raise ValueError("Sem dados suficientes para gerar features. Aumente historico na silver.")

    return X, y


def time_split(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cut = int(len(X) * (1 - test_ratio))
    if cut <= 0 or cut >= len(X):
        raise ValueError("Split temporal inválido. Verifique volume de dados.")

    X_train = X.iloc[:cut]
    X_test = X.iloc[cut:]
    y_train = y.iloc[:cut]
    y_test = y.iloc[cut:]
    return X_train, X_test, y_train, y_test


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
        raise ImportError(
            "Nenhum modelo disponível"
        )

    return models


def run_forecast() -> list[EvalResult]:
    df = load_silver_data()
    step_minutes = infer_step_minutes(df)

    print(f"Passo temporal inferido: {step_minutes} minuto(s)")
    print(f"Linhas disponiveis para treino: {len(df)}")

    horizons = [30, 60, 120, 180]
    models = available_models()

    results: list[EvalResult] = []

    for horizon in horizons:
        try:
            X, y = make_features(df, horizon_minutes=horizon, step_minutes=step_minutes)
            X_train, X_test, y_train, y_test = time_split(X, y, test_ratio=0.2)
        except Exception as exc:
            print(f"[h={horizon}min] pulado: {exc}")
            continue

        print(f"\n[h={horizon}min] train={len(X_train)} test={len(X_test)}")

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                result = EvalResult(
                    horizon_minutes=horizon,
                    model_name=name,
                    mae=mae(y_test.to_numpy(), np.asarray(pred)),
                    rmse=rmse(y_test.to_numpy(), np.asarray(pred)),
                    n_train=len(X_train),
                    n_test=len(X_test),
                )
                results.append(result)
                print(f"  {name:<16} MAE={result.mae:.4f} RMSE={result.rmse:.4f}")
            except Exception as exc:
                print(f"  {name:<16} falhou: {exc}")

    return results


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


def main():
    all_results = run_forecast()
    print_best_by_horizon(all_results)


if __name__ == "__main__":
    main()