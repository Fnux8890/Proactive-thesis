from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.helpers import chain
from airflow.utils.task_group import TaskGroup
import os
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def _check_source_data():
    """Check if new data is available in source directories"""
    input_path = Path('/app/data')  # This should match your mounted volume path
    
    # Define valid file extensions
    valid_extensions = {
        'csv': '.csv',
        'json': '.json',
        'excel': ('.xls', '.xlsx')
    }
    
    # Initialize counters for each data source and file type
    counts = {
        'aarslev': {ext: 0 for ext in valid_extensions.keys()},
        'knudjepsen': {ext: 0 for ext in valid_extensions.keys()}
    }
    
    # Check aarslev directory
    aarslev_path = input_path / 'aarslev'
    if aarslev_path.exists():
        for file in aarslev_path.glob('**/*.*'):
            if file.suffix == valid_extensions['csv']:
                counts['aarslev']['csv'] += 1
            elif file.suffix == valid_extensions['json']:
                counts['aarslev']['json'] += 1
            elif file.suffix in valid_extensions['excel']:
                counts['aarslev']['excel'] += 1
    
    # Check knudjepsen directory
    knudjepsen_path = input_path / 'knudjepsen'
    if knudjepsen_path.exists():
        for file in knudjepsen_path.glob('**/*.*'):
            if file.suffix == valid_extensions['csv']:
                counts['knudjepsen']['csv'] += 1
            elif file.suffix == valid_extensions['json']:
                counts['knudjepsen']['json'] += 1
            elif file.suffix in valid_extensions['excel']:
                counts['knudjepsen']['excel'] += 1
    
    # Calculate totals
    aarslev_total = sum(counts['aarslev'].values())
    knudjepsen_total = sum(counts['knudjepsen'].values())
    
    if aarslev_total == 0 and knudjepsen_total == 0:
        raise ValueError("No valid source files (CSV, JSON, XLS/XLSX) found in input directories")
    
    print(f"Found files in aarslev: {counts['aarslev']}")
    print(f"Found files in knudjepsen: {counts['knudjepsen']}")
        
    return {
        'aarslev': counts['aarslev'],
        'knudjepsen': counts['knudjepsen'],
        'aarslev_total': aarslev_total,
        'knudjepsen_total': knudjepsen_total
    }

def _verify_staging_data(**context):
    """Verify data was loaded into staging correctly"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    # Get source file counts from previous task
    source_counts = context['task_instance'].xcom_pull(task_ids='check_source_data')
    if not source_counts:
        raise ValueError("Could not get source file counts from previous task")
    
    # Check if staging tables exist and have data for each source
    for source in ['aarslev', 'knudjepsen']:
        if source_counts[f'{source}_total'] > 0:
            staging_tables = pg_hook.get_records(f"""
                SELECT table_name, COUNT(*) as record_count
                FROM information_schema.tables t
                JOIN staging.raw_{source} rs ON t.table_name = 'raw_{source}'
                WHERE t.table_schema = 'staging'
                AND rs._status = 'new'
                GROUP BY table_name;
            """)
            
            if not staging_tables:
                raise ValueError(f"No data found in staging tables for {source}")
            
            # Verify record counts
            for table_name, record_count in staging_tables:
                if record_count == 0:
                    raise ValueError(f"No records found in staging table {table_name}")
                print(f"Found {record_count} records in {table_name}")
    
    return True

def _verify_processed_data(**context):
    """Verify processed data exists in TimescaleDB"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    # Get source file counts from check task
    source_counts = context['task_instance'].xcom_pull(task_ids='check_source_data')
    if not source_counts:
        raise ValueError("Could not get source file counts from previous task")
    
    # Check processed data for each source
    for source in ['aarslev', 'knudjepsen']:
        if source_counts[f'{source}_total'] > 0:
            processed_data = pg_hook.get_records(f"""
                SELECT COUNT(*) as record_count
                FROM timeseries.{source}_data
                WHERE _processed_at >= NOW() - INTERVAL '1 hour';
            """)
            
            if not processed_data or processed_data[0][0] == 0:
                raise ValueError(f"No recently processed data found for {source}")
            
            print(f"Found {processed_data[0][0]} recently processed records for {source}")
    
    return True

with DAG(
    'data_ingestion_pipeline',
    default_args=default_args,
    description='Data ingestion pipeline for time series data',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data_ingestion'],
) as dag:
    
    # Check if new data exists
    check_data = PythonOperator(
        task_id='check_source_data',
        python_callable=_check_source_data,
    )
    
    # Data Loading Group
    with TaskGroup(group_id='data_loading') as data_loading:
        load_data = DockerOperator(
            task_id='load_to_staging',
            image='data_processing',
            container_name='data_loader',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
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
        
        # Set dependencies within the group
        load_data >> verify_staging
    
    # Data Processing Group
    with TaskGroup(group_id='data_processing') as data_processing:
        process_data = DockerOperator(
            task_id='process_data',
            image='data_processing',
            container_name='data_processor',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/process_data.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'DATA_INPUT_PATH': '/app/data',
            },
        )
        
        verify_processing = PythonOperator(
            task_id='verify_processing',
            python_callable=_verify_processed_data,
        )
        
        # Set dependencies within the group
        process_data >> verify_processing
    
    # Define the main flow
    chain(
        check_data,
        data_loading,
        data_processing
    ) 