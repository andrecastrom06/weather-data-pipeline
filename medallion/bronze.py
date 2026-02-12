import requests 
import json
from pathlib import Path
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
API_KEY = os.getenv('API_KEY')

BASE_DIR = Path(__file__).resolve().parent.parent

dotenv_path = BASE_DIR / "config" / ".env"

load_dotenv(dotenv_path)

API_KEY = os.getenv("API_KEY")

def bronze() -> dict:
    if not API_KEY:
        logging.error("API_KEY não encontrada. Verifique o arquivo .env")
        return {}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": "Recife,BR",
        "units": "metric",
        "appid": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            try:
                API_KEY = os.getenv('API_SECOND')
            except Exception as e:
                logging.error(f"Erro ao carregar API_SECOND: {e}")
                return {}
            logging.error(
                f"Erro na requisição | Status: {response.status_code} | Resposta: {data}"
            )
            return {}

        if not data:
            logging.warning("Nenhum dado retornado")
            return {}

        output_path = 'data/raw_data.json'
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logging.info(f"Arquivo salvo em {output_path}")
        return data

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro de conexão: {e}")
        return {}

if __name__ == "__main__":
    bronze()