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

def _check_for_new_staging_data(**context):
    """Check for new data in staging tables that needs processing"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    try:
        # Check for new data across all staging tables
        sources = ['aarslev', 'knudjepsen', 'generic']
        source_counts = {}
        total_count = 0
        
        for source in sources:
            # Get count of new records for this source
            result = pg_hook.get_records(f"""
                SELECT COUNT(*) 
                FROM staging.raw_{source}
                WHERE _status = 'new'
            """)
            
            count = result[0][0] if result else 0
            source_counts[source] = count
            total_count += count
        
        if total_count == 0:
            print("No new data found in staging tables")
            return False
        
        # Store counts in XCom for downstream tasks
        context['ti'].xcom_push(key='staging_counts', value=source_counts)
        context['ti'].xcom_push(key='total_staging_count', value=total_count)
        
        print(f"Found {total_count} new records in staging tables: {source_counts}")
        return True
    
    except Exception as e:
        print(f"Error checking for new staging data: {str(e)}")
        return False

def _update_metadata_service(**context):
    """Update the metadata service with information about the processing run"""
    import requests
    
    processing_start = context['execution_date'].isoformat()
    staging_counts = context['ti'].xcom_pull(key='staging_counts') or {}
    
    try:
        # Call the metadata service API to register the processing run
        response = requests.post(
            'http://metadata_service:8000/api/processing/register',
            json={
                'processing_start': processing_start,
                'staging_counts': staging_counts,
                'dag_id': context['dag'].dag_id,
                'dag_run_id': context['dag_run'].run_id,
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"Error updating metadata service: {response.status_code}")
            return False
        
        result = response.json()
        processing_id = result.get('processing_id')
        
        # Store the processing ID for downstream tasks
        if processing_id:
            context['ti'].xcom_push(key='processing_id', value=processing_id)
            print(f"Registered processing run with ID: {processing_id}")
            return True
        else:
            print("No processing ID returned from metadata service")
            return False
    
    except Exception as e:
        print(f"Error updating metadata service: {str(e)}")
        return False

with DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='Process data from staging tables into timeseries tables',
    schedule_interval=None,  # Triggered by the ingestion pipeline
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data_processing'],
) as dag:
    
    # Check for new data in staging tables
    check_staging = PythonOperator(
        task_id='check_for_new_staging_data',
        python_callable=_check_for_new_staging_data,
    )
    
    # Update metadata service about processing run
    update_metadata = PythonOperator(
        task_id='update_metadata_service',
        python_callable=_update_metadata_service,
    )
    
    # Process Aarslev data
    with TaskGroup(group_id='process_aarslev') as process_aarslev:
        # Clean and validate Aarslev data
        clean_aarslev = DockerOperator(
            task_id='clean_aarslev_data',
            image='data_processing',
            container_name='clean_aarslev',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/data_processor.py --source aarslev --action clean',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'PROCESSING_ID': "{{ ti.xcom_pull(key='processing_id') }}",
            },
        )
        
        # Transform Aarslev data from staging to timeseries
        transform_aarslev = DockerOperator(
            task_id='transform_aarslev_data',
            image='data_processing',
            container_name='transform_aarslev',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/data_processor.py --source aarslev --action transform',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'PROCESSING_ID': "{{ ti.xcom_pull(key='processing_id') }}",
            },
        )
        
        # Finalize Aarslev data processing
        finalize_aarslev = PostgresOperator(
            task_id='finalize_aarslev_processing',
            postgres_conn_id='timescaledb',
            sql="""
            UPDATE staging.raw_aarslev
            SET _status = 'processed', 
                _processed_at = NOW()
            WHERE _status = 'validated'
            AND _error_message IS NULL;
            """,
        )
        
        # Set task dependencies
        clean_aarslev >> transform_aarslev >> finalize_aarslev
    
    # Process Knudjepsen data
    with TaskGroup(group_id='process_knudjepsen') as process_knudjepsen:
        # Clean and validate Knudjepsen data
        clean_knudjepsen = DockerOperator(
            task_id='clean_knudjepsen_data',
            image='data_processing',
            container_name='clean_knudjepsen',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/data_processor.py --source knudjepsen --action clean',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'PROCESSING_ID': "{{ ti.xcom_pull(key='processing_id') }}",
            },
        )
        
        # Transform Knudjepsen data from staging to timeseries
        transform_knudjepsen = DockerOperator(
            task_id='transform_knudjepsen_data',
            image='data_processing',
            container_name='transform_knudjepsen',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/data_processor.py --source knudjepsen --action transform',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'PROCESSING_ID': "{{ ti.xcom_pull(key='processing_id') }}",
            },
        )
        
        # Finalize Knudjepsen data processing
        finalize_knudjepsen = PostgresOperator(
            task_id='finalize_knudjepsen_processing',
            postgres_conn_id='timescaledb',
            sql="""
            UPDATE staging.raw_knudjepsen
            SET _status = 'processed', 
                _processed_at = NOW()
            WHERE _status = 'validated'
            AND _error_message IS NULL;
            """,
        )
        
        # Set task dependencies
        clean_knudjepsen >> transform_knudjepsen >> finalize_knudjepsen
    
    # Finalize processing and notify feature extraction
    with TaskGroup(group_id='finalize_processing') as finalize_processing:
        # Update processing statistics
        update_stats = PostgresOperator(
            task_id='update_processing_stats',
            postgres_conn_id='timescaledb',
            sql="""
            INSERT INTO timeseries.processing_runs (
                processing_id, 
                start_time, 
                end_time, 
                records_processed,
                success_rate
            )
            VALUES (
                '{{ ti.xcom_pull(key="processing_id") }}',
                '{{ execution_date.isoformat() }}',
                NOW(),
                (SELECT COUNT(*) FROM staging.raw_aarslev WHERE _status = 'processed' AND _processed_at >= '{{ execution_date.isoformat() }}') +
                (SELECT COUNT(*) FROM staging.raw_knudjepsen WHERE _status = 'processed' AND _processed_at >= '{{ execution_date.isoformat() }}'),
                (
                    SELECT 
                        CASE 
                            WHEN COUNT(*) = 0 THEN 0
                            ELSE COUNT(*) FILTER (WHERE _status = 'processed') * 100.0 / COUNT(*)
                        END
                    FROM (
                        SELECT _status FROM staging.raw_aarslev WHERE _processed_at >= '{{ execution_date.isoformat() }}'
                        UNION ALL
                        SELECT _status FROM staging.raw_knudjepsen WHERE _processed_at >= '{{ execution_date.isoformat() }}'
                    ) combined
                )
            );
            """,
        )
        
        # Notify metadata service that processing is complete
        notify_metadata = SimpleHttpOperator(
            task_id='notify_metadata_service',
            http_conn_id='metadata_service',
            endpoint='/api/processing/complete',
            method='POST',
            data=json.dumps({
                "processing_id": "{{ ti.xcom_pull(key='processing_id') }}",
                "completion_time": "{{ ts }}"
            }),
            headers={"Content-Type": "application/json"},
        )
        
        # Trigger feature extraction DAG
        trigger_feature_extraction = PythonOperator(
            task_id='trigger_feature_extraction',
            python_callable=lambda **kwargs: kwargs['ti'].xcom_push(key='should_extract_features', value=True),
        )
        
        # Set task dependencies
        update_stats >> notify_metadata >> trigger_feature_extraction
    
    # Define the overall DAG structure
    chain(
        check_staging,
        update_metadata,
        [process_aarslev, process_knudjepsen],
        finalize_processing
    ) 