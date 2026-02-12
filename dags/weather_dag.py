from datetime import timedelta, datetime
from airflow.decorators import dag, task
import sys

sys.path.insert(0, '/opt/airflow')

from medallion.bronze import bronze
from medallion.silver import silver

@dag(
    dag_id='weather_pipeline',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    },
    description='Pipeline clima Olinda',
    schedule='0 * * * *',
    start_date=datetime(2026,2,11),
    catchup=False,
    tags=['weather', 'etl', 'olinda']
)

def dag_weather():
    @task
    def bronze_layer():
        bronze()
    
    @task
    def silver_layer():
        silver()

    bronze_layer() >> silver_layer()

dag_weather()