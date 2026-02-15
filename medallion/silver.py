import pandas as pd
from pathlib import Path
import json
import logging
from medallion.utils.connection import engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_df() -> pd.DataFrame:
    path = Path(__file__).parent.parent / 'data' / 'bronze_olinda_weather.json'

    if not path.exists():
        raise FileNotFoundError("Arquivo bronze_olinda_weather.json não encontrado na camada Bronze")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar JSON: {e}")

    if not data:
        raise ValueError("JSON vazio ou inválido")

    df = pd.json_normalize(data)

    logging.info("DataFrame criado corretamente!")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tratamento dos dados")

    columns_to_drop = {
        'weather',
        'weather_icon',
        'sys.type'
    }

    columns_to_rename = {
        'dt': 'datetime',
        'cod': 'code',
        'coord.lon': 'longitude',
        'coord.lat': 'latitude',
        'main.temp': 'temperature',
        'main.feels_like': 'feels_like',
        'main.temp_min': 'temperature_min',
        'main.temp_max': 'temperature_max',
        'main.pressure': 'pressure',
        'main.humidity': 'humidity',
        'main.sea_level': 'sea_level',
        'main.grnd_level': 'grand_level',
        'wind.speed': 'wind_speed',
        'wind.deg': 'wind_deg',
        'rain.1h': 'rain_last_1h',
        'clouds.all': 'clouds',
        'sys.id': 'sys_id',
        'sys.country': 'sys_country',
        'sys.sunrise': 'sunrise',
        'sys.sunset': 'sunset'
    }

    datetime_columns = {'datetime', 'sunrise', 'sunset'}

    try:
        if 'weather' not in df.columns:
            raise KeyError("Coluna 'weather' não encontrada no DataFrame")

        df_weather = pd.json_normalize(df['weather'].apply(lambda x: x[0]))
        df_weather = df_weather.rename(columns={
            'id': 'weather_id',
            'main': 'weather_main',
            'description': 'weather_description',
            'icon': 'weather_icon'
        })

        df = pd.concat([df, df_weather], axis=1)

        df = df.drop(columns=columns_to_drop, errors='ignore')

        df = df.rename(columns=columns_to_rename)

        for c in datetime_columns:
            if c in df.columns:
                df[c] = (
                    pd.to_datetime(df[c], unit='s', utc=True)
                    .dt.tz_convert('America/Sao_Paulo')
                )

    except Exception as e:
        raise RuntimeError(f"Erro durante transformação da camada Silver: {e}")

    logging.info("Transformações concluídas com sucesso")
    return df


def load_bd(df: pd.DataFrame):
    logging.info("Conectando ao banco de dados")

    df.to_sql(
        "silver_olinda_weather",
        engine,
        if_exists="append",  
        index=False,
        method="multi"       
    )

    logging.info("Dados inseridos com sucesso na camada Silver")


def silver():
    df = create_df()
    df_normalizado = normalize_columns(df)
    load_bd(df_normalizado)