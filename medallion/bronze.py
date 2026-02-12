import requests
import json
from pathlib import Path
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / "config" / ".env"
load_dotenv(dotenv_path)

API_KEY = os.getenv('API_KEY')


def bronze() -> dict:
    if not API_KEY:
        raise ValueError("API_KEY não encontrada. Verifique o arquivo .env")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": "Olinda,BR",
        "units": "metric",
        "appid": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data:
            raise ValueError("Nenhum dado retornado pela API")

        output_path = Path("data/raw_data.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logging.info(f"Arquivo salvo em {output_path}")
        return data

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Erro na requisição para OpenWeather: {e}")