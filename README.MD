# Weather Data Pipeline (Olinda/PE)

Pipeline de dados climáticos com arquitetura **Medallion (Bronze -> Silver -> Gold)**, orquestrado com **Apache Airflow** e persistido em **PostgreSQL**.

Este projeto coleta dados atuais de clima da API **OpenWeatherMap** para a cidade de **Olinda/BR**, normaliza o payload JSON e grava os dados tratados em tabela relacional para análise. Finalizando com **Machine Learning** para fazer a previsão da temperatura nos próximos minutos ou horas.

---

## Tecnologias utilizadas

- **Python 3.12+**
- **Apache Airflow 3 (CeleryExecutor)**
- **Docker + Docker Compose**
- **PostgreSQL 16**
- **Redis** (Broker do Celery)
- **Pandas**
- **NumPy**
- **SQLAlchemy**
- **Requests**
- **python-dotenv**
- **pyarrow**
- **scikit-learn**
- **XGBoost**
- **LightGBM**
- **Arquitetura Medallion**
- **AWS**
- **Neon DB Console**

---

## Arquitetura do pipeline

### Bronze (Extração)
Arquivo: `medallion/bronze.py`

- Consome endpoint `/data/2.5/weather` da OpenWeatherMap.
- Busca clima de `Olinda,BR` em unidade métrica.
- Salva o JSON bruto em:
  - `data/bronze_olinda_weather.json`

### Silver (Transformação + Carga)
Arquivo: `medallion/silver.py`

- Lê o JSON da camada bronze.
- Normaliza colunas aninhadas.
- Renomeia campos para padrão mais analítico.
- Converte timestamps para timezone `America/Sao_Paulo`.
- Grava no Postgres em:
  - `silver_olinda_weather`

### Gold (Agregação de valores analíticos)
Arquivo: `medallion/gold.py`

- Lê os dados tratados da camada Silver (`silver_olinda_weather`).
- Converte datetime para tipo data.
- Agrupa os dados por:
  - `date`
  - `sys_country`
- Calcula métricas agregadas diárias:
  - `temperature_min`, `temperature_avg`, `temperature_max`
  - `humidity_avg`
  - `pressure_avg`
  - `wind_speed_avg`
- Identifica o clima predominante do dia:
  - `dominant_weather_main`
  - `dominant_weather_description`
- Grava no Postgres em:
  - `gold_olinda_weather_daily`

### Orquestração (Airflow)
Arquivo: `dags/weather_dag.py`

- DAG: `weather_pipeline`
- Frequência: **a cada 5 minutos** (`*/5 * * * *`)
- Ordem das tasks:
  1. `bronze_layer`
  2. `silver_layer`
  3. `gold_layer`

---

## Machine Learning (Previsão de temperatura)

Arquivo: `ml/preview_temperature.py`

Foi implementado um fluxo de **previsão de temperatura de curto prazo (30, 60, 120 e 180 minutos)** com regressão de serie temporal.

### O que o script faz

- Lê dados historicos da tabela `silver_olinda_weather`.
- Cria variáveis de entrada com:
  - Lags de `temperature`, `humidity`, `pressure`, `wind_speed`
  - Hora do dia (`hour`)
  - Dia da semana (`day_of_week`)
- Cria o alvo `target_temperature` para cada horizonte futuro.
- Faz split temporal treino/teste (sem embaralhar).
- Treina e compara os modelos:
  - `LinearRegression` (baseline)
  - `XGBoost`
  - `LightGBM`
- Avalia com metricas:
  - `MAE` Média do erro absoluto: media(|real - previsto|).
  - `RMSE` Raiz da média do erro ao quadrado: sqrt(media((real - previsto)^2)).
- Exibe o melhor modelo por horizonte (menor RMSE).

### Como executar

```bash
python ml/preview_temperature.py
```

> Observação: para ter previsões estáveis, a tabela `silver_olinda_weather` precisa de histórico suficiente (na prática, várias coletas ao longo do dia).

---

## Estrutura do projeto

```bash
.
|-- dags/
|   |-- weather_dag.py
|-- data/
|   |-- bronze_olinda_weather.json
|-- medallion/
|   |-- bronze.py
|   |-- silver.py
|   |-- gold.py
|   -- utils/
|       -- connection.py
|-- ml/
|   -- preview_temperature.py
|-- docker-compose.yaml
|-- main.py
|-- requirements.txt
-- README.MD
```

---

## Configuração de ambiente

Crie o arquivo `config/.env` (o código lê variáveis dessa localização) com:

```env
# API
API_KEY=sua_chave_openweathermap

# Banco de dados
DB_USER=Usuário
DB_PASSWORD=Senha
DB_HOST=Host
DB_PORT=5432
DB_NAME=Nome do banco
```

> Observação: para execução local sem Docker, ajuste `DB_HOST` para `localhost` (ou host equivalente).

---

## Banco de Dados

Utiliza banco de dados em núvem, com serviço da AWS pelo Neon DB Console. Utilizamos PostgreSQL versão 17 na região US East 1 (N. Virginia)

---

## Como executar com Docker Compose

### 1) Subir os serviços

```bash
docker compose up -d
```

Para parar os servicos:

```bash
docker compose down
```

### 2) Inicializar/validar containers

```bash
docker compose ps
```

### 3) Acessar interface do Airflow

- URL: `http://localhost:8080`

Ative a DAG `weather_pipeline` e execute manualmente (ou aguarde o agendamento).

---

## Execução manual (sem Airflow)

Você tambem pode disparar Bronze + Silver + Gold diretamente:

```bash
python main.py
```

---

## Consultando os dados no PostgreSQL

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

## Campos esperados na camada Silver

Alguns campos gerados apos normalização:

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

## Dependencias Python

Instalação (execução local):

```bash
pip install -r requirements.txt
```

Dependências principais declaradas no projeto:

- `pandas`
- `numpy`
- `requests`
- `python-dotenv`
- `psycopg2-binary`
- `sqlalchemy`
- `pyarrow`
- `scikit-learn`
- `xgboost`
- `lightgbm`

---

## Pontos de atenção

- Sem `API_KEY`, a camada Bronze falha por validação.
- Sem variáveis de banco (`DB_*`), a conexão SQLAlchemy não é criada.
- A DAG esta com `start_date` fixo em 2026 no código; ajuste se quiser retroagir/agendar diferente.
- Com poucas linhas na `silver_olinda_weather`, o script de ML pode pular alguns horizontes por falta de histórico.

---