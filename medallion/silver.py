import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_df() -> pd.DataFrame:
    path = Path(__file__).parent.parent / 'data' / 'raw_data.json'
    if not path.exists():
        raise FileNotFoundError("Arquivo não encontrado")
        
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Erro no JSON: {e}")
        raise

    if not data:
        raise ValueError("JSON vazio ou inválido")

    df = pd.json_normalize(data)
    logging.info(f"DataFrame criado corretamente!")
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Iniciando tratamento dos dados")
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
        'clouds.all': 'clouds', 
        'sys.id': 'sys_id', 
        'sys.country': 'sys_country',
        'sys.sunrise': 'sunrise', 
        'sys.sunset': 'sunset'
    }

    datetime_columns = {
        'datetime', 'sunrise', 'sunset'
    }

    df_weather = pd.json_normalize(df['weather'].apply(lambda x: x[0]))
    df_weather= df_weather.rename(columns={
        'id': 'weather_id',
        'main': 'weather_main',
        'description': 'weather_description',
        'icon': 'weather_icon'
    })

    df = pd.concat([df, df_weather], axis=1)

    df = df.drop(columns=columns_to_drop)
    df = df.rename(columns=columns_to_rename)
    for c in datetime_columns:
        df[c] = pd.to_datetime(df[c], unit='s', utc=True).dt.tz_convert('America/Sao_Paulo')
    return df

def json_to_parquet(df: pd.DataFrame):
    logging.info(f"Tranformando em arquivo parquet")

    path = Path(__file__).parent.parent
    output_path = path / "data" / "transformed_data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(
            output_path,
            engine="pyarrow",
            index=False
        )

        logging.info(f"Arquivo parquet salvo!")

    except Exception as e:
        logging.error(f"Erro ao salvar parquet: {e}")
        raise

def silver():
    df = create_df()
    df_normalizado = normalize_columns(df)
    json_to_parquet(df_normalizado)

if __name__ == '__main__':
    silver()