from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.helpers import chain

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def _check_data_availability():
    """Check if new data is available for processing"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    # Check for new files in the input directory
    # This will be implemented based on your specific requirements
    return True

def _verify_staging_data():
    """Verify data was loaded into staging correctly"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    # Check if staging tables exist and have data
    result = pg_hook.get_records("""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = 'staging'
    """)
    if result and result[0][0] > 0:
        # Check for records with 'new' status
        result = pg_hook.get_records("""
            SELECT COUNT(*) 
            FROM staging.raw_aarslev 
            WHERE _status = 'new'
        """)
        return result and result[0][0] > 0
    return False

def _post_process_check():
    """Verify data was processed and stored correctly"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    try:
        # Check if processed data exists
        result = pg_hook.get_records("""
            SELECT COUNT(*) 
            FROM processed_data_aarslev
        """)
        return result and result[0][0] > 0
    except Exception as e:
        print(f"Error checking processed data: {e}")
        return False

with DAG(
    'data_ingestion_pipeline',
    default_args=default_args,
    description='Data ingestion pipeline for time series data',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data_ingestion'],
) as dag:

    check_data = PythonOperator(
        task_id='check_data_availability',
        python_callable=_check_data_availability,
    )

    load_data = DockerOperator(
        task_id='load_data',
        image='data_processing',
        container_name='data_loader',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='data-ingestion_network',  # Use the network defined in docker-compose
        command='uv run /app/data_loader.py',
        environment={
            'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
            'DATA_INPUT_PATH': '/app/data',
        },
    )

    verify_staging = PythonOperator(
        task_id='verify_staging',
        python_callable=_verify_staging_data,
    )

    process_data = DockerOperator(
        task_id='process_data',
        image='data_processing',
        container_name='data_processor',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='data-ingestion_network',  # Use the network defined in docker-compose
        command='uv run /app/process_data.py',
        environment={
            'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
            'DATA_INPUT_PATH': '/app/data',
        },
    )

    verify_processing = PythonOperator(
        task_id='verify_processing',
        python_callable=_post_process_check,
    )

    # Define the task chain
    chain(
        check_data,
        load_data,
        verify_staging,
        process_data,
        verify_processing
    ) 