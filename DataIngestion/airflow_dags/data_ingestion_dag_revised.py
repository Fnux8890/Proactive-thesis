from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.helpers import chain
from airflow.utils.task_group import TaskGroup
import json
import os
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def _check_for_new_files(**context):
    """Check for new files by calling the Elixir ingestion API"""
    # The Elixir API handles file discovery and registration
    # This function just checks if there are new files to process
    
    import requests
    
    # Call the Elixir API to get status of pending files
    try:
        response = requests.get(
            'http://elixir_ingestion:4000/api/files/list?status=pending&limit=100',
            timeout=10
        )
        
        if response.status_code != 200:
            raise ValueError(f"API request failed with status code {response.status_code}")
        
        data = response.json()
        files = data.get('files', [])
        
        if not files:
            print("No new files to process")
            return False
        
        # Store file info in XCom for downstream tasks
        context['ti'].xcom_push(key='files_to_process', value=files)
        
        # Count files by source
        file_counts = {}
        for file in files:
            source = file.get('source', 'unknown')
            if source not in file_counts:
                file_counts[source] = 0
            file_counts[source] += 1
        
        print(f"Found {len(files)} new files to process: {file_counts}")
        return True
        
    except Exception as e:
        print(f"Error checking for new files: {str(e)}")
        return False

def _load_files_to_staging(**context):
    """Prepare SQL statements to load files into staging tables"""
    # Get files from XCom
    files = context['ti'].xcom_pull(key='files_to_process')
    
    if not files:
        print("No files to load to staging")
        return []
    
    # Group files by source
    files_by_source = {}
    for file in files:
        source = file.get('source', 'unknown')
        if source not in files_by_source:
            files_by_source[source] = []
        files_by_source[source].append(file)
    
    # Process each source separately
    for source, source_files in files_by_source.items():
        # Call the Python loading script for each source
        # The script is responsible for connecting to the database and loading the data
        try:
            # Push a list of file paths for each source to load
            file_paths = [f['file_path'] for f in source_files]
            context['ti'].xcom_push(key=f'files_to_load_{source}', value=file_paths)
            
            print(f"Prepared {len(file_paths)} files for loading from source {source}")
        except Exception as e:
            print(f"Error preparing files for {source}: {str(e)}")
    
    return True

def _verify_staging_data(**context):
    """Verify that data was successfully loaded into staging tables"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    # Get files from XCom
    files = context['ti'].xcom_pull(key='files_to_process')
    
    if not files:
        print("No files to verify in staging")
        return False
    
    # Check if records were loaded for each file
    for file in files:
        source = file.get('source', 'unknown')
        file_path = file.get('file_path', '')
        
        if not file_path:
            continue
        
        try:
            # Query the staging table for records from this file
            result = pg_hook.get_records(f"""
                SELECT COUNT(*) 
                FROM staging.raw_{source}
                WHERE source_file = %s
                AND _status = 'new'
            """, (file_path,))
            
            if not result or result[0][0] == 0:
                print(f"No records found in staging for file {file_path}")
                
                # Update file status in tracking table
                pg_hook.run(f"""
                    UPDATE staging.ingestion_files
                    SET status = 'error',
                        error_message = 'No records loaded into staging'
                    WHERE file_path = %s
                """, (file_path,))
            else:
                record_count = result[0][0]
                print(f"Found {record_count} records in staging for file {file_path}")
                
                # Update file status in tracking table
                pg_hook.run(f"""
                    UPDATE staging.ingestion_files
                    SET status = 'loaded',
                        record_count = %s,
                        ingestion_completed_at = NOW()
                    WHERE file_path = %s
                """, (record_count, file_path))
                
        except Exception as e:
            print(f"Error verifying staging data for file {file_path}: {str(e)}")
    
    # Check overall results
    result = pg_hook.get_records("""
        SELECT COUNT(*) FROM staging.ingestion_files
        WHERE status = 'loaded'
        AND ingestion_completed_at >= NOW() - INTERVAL '1 hour'
    """)
    
    if result and result[0][0] > 0:
        return True
    else:
        return False

def _notify_data_processing_dag(**context):
    """Trigger the data processing DAG to transform staged data into timeseries tables"""
    from airflow.api.client.local_client import Client
    
    client = Client(None, None)
    
    # Create a configuration with runtime parameters
    execution_date = datetime.utcnow().isoformat()
    conf = {"execution_date": execution_date}
    
    # Trigger the data processing DAG
    try:
        client.trigger_dag(dag_id="data_processing_pipeline", conf=conf)
        print(f"Triggered data processing pipeline with config: {conf}")
        return True
    except Exception as e:
        print(f"Error triggering data processing pipeline: {str(e)}")
        return False

with DAG(
    'data_ingestion_pipeline_revised',
    default_args=default_args,
    description='Revised data ingestion pipeline with clear boundaries',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data_ingestion', 'revised'],
) as dag:
    
    # Check for new files via the Elixir API
    check_files = PythonOperator(
        task_id='check_for_new_files',
        python_callable=_check_for_new_files,
    )
    
    # Create data loading group
    with TaskGroup(group_id='data_loading') as data_loading:
        # Prepare SQL statements for loading files
        prepare_loading = PythonOperator(
            task_id='prepare_loading',
            python_callable=_load_files_to_staging,
        )
        
        # Load Aarslev data to staging
        load_aarslev = DockerOperator(
            task_id='load_aarslev_to_staging',
            image='data_processing',
            container_name='data_loader_aarslev',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/data_loader.py --source aarslev',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'FILES_TO_LOAD': "{{ ti.xcom_pull(key='files_to_load_aarslev') }}",
            },
        )
        
        # Load Knudjepsen data to staging
        load_knudjepsen = DockerOperator(
            task_id='load_knudjepsen_to_staging',
            image='data_processing',
            container_name='data_loader_knudjepsen',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/data_loader.py --source knudjepsen',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'FILES_TO_LOAD': "{{ ti.xcom_pull(key='files_to_load_knudjepsen') }}",
            },
        )
        
        # Verify staged data
        verify_staging = PythonOperator(
            task_id='verify_staging_data',
            python_callable=_verify_staging_data,
        )
        
        # Define task dependencies in the loading group
        prepare_loading >> [load_aarslev, load_knudjepsen] >> verify_staging
    
    # Notify the Elixir API that ingestion is complete
    notify_api = SimpleHttpOperator(
        task_id='notify_ingestion_api',
        http_conn_id='elixir_api',
        endpoint='/api/pipeline/status',
        method='POST',
        data=json.dumps({
            "status": "ingestion_complete",
            "timestamp": "{{ ts }}"
        }),
        headers={"Content-Type": "application/json"},
    )
    
    # Notify the data processing DAG to start processing
    trigger_processing = PythonOperator(
        task_id='trigger_data_processing',
        python_callable=_notify_data_processing_dag,
    )
    
    # Set up task dependencies for the entire DAG
    chain(
        check_files,
        data_loading,
        notify_api,
        trigger_processing
    ) 