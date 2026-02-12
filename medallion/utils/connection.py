from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus

BASE_DIR = Path(__file__).resolve().parent.parent.parent
dotenv_path = BASE_DIR / "config" / ".env"
load_dotenv(dotenv_path)

user = os.getenv("DB_USER")
password_raw = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

if not all([user, password_raw, host, port, db]):
    raise RuntimeError("❌ Variáveis de banco não carregadas do .env")

password = quote_plus(password_raw)

DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

def create_db_engine():
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=180,
            pool_size=10,
            max_overflow=5
        )
        return engine

    except ImportError:
        raise ImportError(
            "❌ Driver não encontrado. Rode: pip install psycopg2-binary"
        )

    except Exception as e:
        raise RuntimeError(f"❌ Erro ao criar engine: {e}")

engine = create_db_engine()
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()