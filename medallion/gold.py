import logging
import pandas as pd
from medallion.utils.connection import engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_silver_data() -> pd.DataFrame:
    query = """
        SELECT
            datetime,
            sys_country,
            temperature,
            humidity,
            pressure,
            wind_speed,
            weather_main,
            weather_description
        FROM silver_olinda_weather
    """

    try:
        df = pd.read_sql(query, con=engine)
    except Exception as e:
        raise RuntimeError(f"Erro ao ler camada Silver: {e}")

    if df.empty:
        raise ValueError("Nenhum dado disponível na tabela silver_olinda_weather")

    logging.info("Dados da Silver carregados com sucesso")
    return df


def build_daily_gold(df: pd.DataFrame) -> pd.DataFrame:
    if 'datetime' not in df.columns:
        raise KeyError("Coluna 'datetime' não encontrada na Silver")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    agg_df = (
        df.groupby(['date', 'sys_country'], as_index=False)
        .agg(
            temperature_min=('temperature', 'min'),
            temperature_avg=('temperature', 'mean'),
            temperature_max=('temperature', 'max'),
            humidity_avg=('humidity', 'mean'),
            pressure_avg=('pressure', 'mean'),
            wind_speed_avg=('wind_speed', 'mean')
        )
    )

    dominant_weather = (
        df.groupby(['date', 'sys_country'])['weather_main']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index(name='dominant_weather_main')
    )

    dominant_description = (
        df.groupby(['date', 'sys_country'])['weather_description']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index(name='dominant_weather_description')
    )

    gold_df = agg_df.merge(dominant_weather, on=['date', 'sys_country'], how='left')
    gold_df = gold_df.merge(dominant_description, on=['date', 'sys_country'], how='left')

    logging.info("Camada Gold construída com sucesso")
    return gold_df


def load_gold(df: pd.DataFrame):
    try:
        df.to_sql(
            name='gold_olinda_weather_daily',
            con=engine,
            if_exists='append',
            index=False,
            method='multi'
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar camada Gold: {e}")

    logging.info("Dados da Gold enviados para o banco com sucesso")


def gold() -> pd.DataFrame:
    silver_df = extract_silver_data()
    gold_df = build_daily_gold(silver_df)
    load_gold(gold_df)
    return gold_df