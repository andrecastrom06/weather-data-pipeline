# 🌦️ Weather Data Pipeline (Olinda/PE)

Pipeline de dados climáticos com arquitetura **Medallion (Bronze → Silver → Gold)**, orquestrado com **Apache Airflow** e persistido em **PostgreSQL**.

Este projeto coleta dados atuais de clima da API **OpenWeatherMap** para a cidade de **Olinda/BR**, normaliza o payload JSON e grava os dados tratados em tabela relacional para análise.

---

## ✅ Tecnologias utilizadas

- **Python 3.12+**
- **Apache Airflow 3 (CeleryExecutor)**
- **Docker + Docker Compose**
- **PostgreSQL 16**
- **Redis** (broker do Celery)
- **Pandas**
- **SQLAlchemy**
- **Requests**
- **python-dotenv**
- **Arquitetura Medallion**

---

## 🧱 Arquitetura do pipeline

### Bronze (Extração)
Arquivo: `medallion/bronze.py`

- Consome endpoint `/data/2.5/weather` da OpenWeatherMap.
- Busca clima de `Olinda,BR` em unidade métrica.
- Salva o JSON bruto em:
  - `data/bronze_olinda_weather.json`

### Silver (Transformação + Carga)
Arquivo: `medallion/silver.py`

- Lê o JSON da camada bronze.
- Normaliza colunas aninhadas (inclusive `weather`).
- Renomeia campos para padrão mais analítico.
- Converte timestamps para timezone `America/Sao_Paulo`.
- Grava no Postgres em:
  - `silver_olinda_weather`

### Gold (Agregação de valores analíticos)
Arquivo: `medallion/gold.py`

- Lê os dados tratados da camada Silver (silver_olinda_weather).
- Converte datetime para tipo data.
- Agrupa os dados por:
    `date`
    `sys_country`
- Calcula métricas agregadas diárias:
    `Temperatura mínima, média e máxima.`
    `Umidade média.`
    `Pressão média.`
    `Velocidade média do vento.`
- Identifica o clima predominante do dia:
    `dominant_weather_main`
    `dominant_weather_description`
- Grava no Postgres em:
    `gold_olinda_weather_daily`

### Orquestração (Airflow)
Arquivo: `dags/weather_dag.py`

- DAG: `weather_pipeline`
- Frequência: **a cada 5 minutos** (`*/5 * * * *`)
- Ordem das tasks:
  1. `bronze_layer`
  2. `silver_layer`
  3. `gold_layer`

---

## 📁 Estrutura do projeto

```bash
.
├── dags/
│   └── weather_dag.py
├── data/
│   └── bronze_olinda_weather.json
├── medallion/
│   ├── bronze.py
│   ├── silver.py
│   ├── gold.py
│   └── utils/
│       └── connection.py
├── docker-compose.yaml
├── main.py
├── requirements.txt
└── README.MD
```

---

## ⚙️ Configuração de ambiente

Crie o arquivo `config/.env` (o código lê variáveis dessa localização) com:

```env
# API
API_KEY=sua_chave_openweathermap

# Banco de dados
DB_USER=airflow
DB_PASSWORD=airflow
DB_HOST=postgres
DB_PORT=5432
DB_NAME=airflow
```

> Observação: para execução local sem Docker, ajuste `DB_HOST` para `localhost` (ou host equivalente).

---

## 🚀 Como executar com Docker Compose

### 1) Subir os serviços

```bash
docker compose up -d
```

### 2) Inicializar/validar containers

```bash
docker compose ps
```

### 3) Acessar interface do Airflow

- URL: `http://localhost:8080`

Ative a DAG `weather_pipeline` e execute manualmente (ou aguarde o agendamento horário).

---

## ▶️ Execução manual (sem Airflow)

Você também pode disparar Bronze + Silver diretamente:

```bash
python main.py
```

---

## 🗃️ Consultando os dados no PostgreSQL

Entrar no container do Postgres:

```bash
docker exec -it etl-postgres-1 psql -U airflow -d airflow
```

Consulta rápida:

```sql
SELECT *
FROM silver_olinda_weather
ORDER BY datetime DESC
LIMIT 10;
```

```sql
SELECT *
FROM gold_olinda_weather_daily
ORDER BY date DESC
LIMIT 10;
```

---

## 🔍 Campos esperados na camada Silver

Alguns campos gerados após normalização:

- `datetime`
- `city`
- `country`
- `temperature`
- `feels_like`
- `humidity`
- `pressure`
- `wind_speed`
- `weather_main`
- `weather_description`
- `sunrise`
- `sunset`

---

## 🧪 Dependências Python

Instalação (execução local):

```bash
pip install -r requirements.txt
```

Dependências principais declaradas no projeto:

- `pandas`
- `requests`
- `python-dotenv`
- `psycopg2-binary`
- `sqlalchemy`
- `pyarrow`

---

## ⚠️ Pontos de atenção

- Sem `API_KEY`, a camada Bronze falha por validação.
- Sem variáveis de banco (`DB_*`), a conexão SQLAlchemy não é criada.
- A DAG está com `start_date` fixo em 2026 no código; ajuste se quiser retroagir/agendar diferente.

---
