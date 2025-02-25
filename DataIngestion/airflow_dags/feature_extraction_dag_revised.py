from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.helpers import chain
from airflow.utils.task_group import TaskGroup
import json
import uuid

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def _check_timeseries_data_readiness(**context):
    """Check if there is new timeseries data ready for feature extraction"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    # Check for new data in timeseries tables that hasn't had features extracted yet
    sources = ['aarslev', 'knudjepsen']
    source_counts = {}
    total_count = 0
    
    for source in sources:
        # Query for unprocessed timeseries data
        result = pg_hook.get_records(f"""
            SELECT COUNT(*) 
            FROM timeseries.{source}_data 
            WHERE _feature_extracted = false
            AND _processed_at IS NOT NULL
        """)
        
        count = result[0][0] if result else 0
        source_counts[source] = count
        total_count += count
    
    if total_count == 0:
        print("No new timeseries data found for feature extraction")
        return False
    
    # Store counts in XCom for downstream tasks
    context['ti'].xcom_push(key='timeseries_counts', value=source_counts)
    context['ti'].xcom_push(key='total_timeseries_count', value=total_count)
    
    # Generate a unique run ID for this extraction run
    run_id = f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    context['ti'].xcom_push(key='feature_extraction_run_id', value=run_id)
    
    print(f"Found {total_count} new records in timeseries tables for feature extraction: {source_counts}")
    print(f"Feature extraction run ID: {run_id}")
    return True

def _register_extraction_run(**context):
    """Register a new feature extraction run in the database"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    run_id = context['ti'].xcom_pull(key='feature_extraction_run_id')
    timeseries_counts = context['ti'].xcom_pull(key='timeseries_counts') or {}
    
    # Convert the counts dictionary to a string array for PostgreSQL
    data_sources = [f"{source}:{count}" for source, count in timeseries_counts.items()]
    
    # Insert the extraction run record
    pg_hook.run("""
        INSERT INTO features.extraction_runs (
            run_id, 
            start_time, 
            status, 
            data_sources,
            time_range_start,
            time_range_end,
            configuration
        )
        VALUES (
            %s, 
            NOW(), 
            'running',
            %s,
            (
                SELECT MIN(time) 
                FROM (
                    SELECT time FROM timeseries.aarslev_data 
                    WHERE _feature_extracted = false
                    UNION ALL
                    SELECT time FROM timeseries.knudjepsen_data 
                    WHERE _feature_extracted = false
                ) as combined
            ),
            (
                SELECT MAX(time) 
                FROM (
                    SELECT time FROM timeseries.aarslev_data 
                    WHERE _feature_extracted = false
                    UNION ALL
                    SELECT time FROM timeseries.knudjepsen_data 
                    WHERE _feature_extracted = false
                ) as combined
            ),
            %s
        )
    """, (
        run_id, 
        data_sources, 
        json.dumps({
            "execution_date": context['execution_date'].isoformat(),
            "dag_id": context['dag'].dag_id,
            "dag_run_id": context['dag_run'].run_id
        })
    ))
    
    print(f"Registered feature extraction run with ID: {run_id}")
    return True

def _update_extraction_run_status(**context):
    """Update the status of the feature extraction run"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    run_id = context['ti'].xcom_pull(key='feature_extraction_run_id')
    
    # Get feature counts
    result = pg_hook.get_records("""
        SELECT COUNT(*) 
        FROM features.extracted_features
        WHERE created_at >= (
            SELECT start_time 
            FROM features.extraction_runs 
            WHERE run_id = %s
        )
    """, (run_id,))
    
    features_count = result[0][0] if result else 0
    
    # Update the extraction run record
    pg_hook.run("""
        UPDATE features.extraction_runs
        SET 
            status = 'completed',
            end_time = NOW(),
            features_count = %s,
            processing_time_ms = EXTRACT(EPOCH FROM (NOW() - start_time)) * 1000
        WHERE run_id = %s
    """, (features_count, run_id))
    
    print(f"Updated feature extraction run {run_id} with {features_count} features")
    return True

def _log_feature_stats(**context):
    """Log statistics about extracted features"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    run_id = context['ti'].xcom_pull(key='feature_extraction_run_id')
    
    # Get counts of extracted features by source
    feature_counts = pg_hook.get_records("""
        SELECT source, COUNT(DISTINCT feature_name) as feature_count
        FROM features.extracted_features
        WHERE created_at >= (
            SELECT start_time 
            FROM features.extraction_runs 
            WHERE run_id = %s
        )
        GROUP BY source
    """, (run_id,))
    
    if not feature_counts:
        print("No features were extracted in this run")
        return False
    
    for source, count in feature_counts:
        print(f"Extracted {count} features for {source}")
    
    # Get the top features by importance score
    top_features = pg_hook.get_records("""
        SELECT fi.source, fi.feature_name, fi.importance_score, fm.description
        FROM features.feature_importance fi
        LEFT JOIN features.feature_metadata fm ON fi.feature_name = fm.feature_name
        WHERE fi.created_at >= (
            SELECT start_time 
            FROM features.extraction_runs 
            WHERE run_id = %s
        )
        ORDER BY fi.importance_score DESC
        LIMIT 10
    """, (run_id,))
    
    print("Top 10 features by importance score:")
    for source, feature_name, score, description in top_features:
        desc = description or "No description available"
        print(f"{source} - {feature_name}: {score} ({desc})")
    
    return True

with DAG(
    'feature_extraction_pipeline_revised',
    default_args=default_args,
    description='Revised feature extraction pipeline that reads exclusively from TimescaleDB',
    schedule_interval=None,  # Triggered by the data processing pipeline
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['features', 'data_science', 'revised'],
) as dag:
    
    # Check if new timeseries data exists for feature extraction
    check_timeseries_data = PythonOperator(
        task_id='check_timeseries_data_readiness',
        python_callable=_check_timeseries_data_readiness,
    )
    
    # Register the feature extraction run
    register_extraction_run = PythonOperator(
        task_id='register_extraction_run',
        python_callable=_register_extraction_run,
    )
    
    # Feature Discovery Group
    with TaskGroup(group_id='feature_discovery') as feature_discovery:
        # Extract statistical features
        extract_statistical_features = DockerOperator(
            task_id='extract_statistical_features',
            image='feature_extraction',
            container_name='statistical_features',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/feature_extractors/statistical_features.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'EXTRACTION_RUN_ID': "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            },
        )
        
        # Extract temporal features
        extract_temporal_features = DockerOperator(
            task_id='extract_temporal_features',
            image='feature_extraction',
            container_name='temporal_features',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/feature_extractors/temporal_features.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'EXTRACTION_RUN_ID': "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            },
        )
        
        # Extract domain-specific features
        extract_domain_features = DockerOperator(
            task_id='extract_domain_features',
            image='feature_extraction',
            container_name='domain_features',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/feature_extractors/domain_features.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'EXTRACTION_RUN_ID': "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            },
        )
    
    # Feature Evaluation Group
    with TaskGroup(group_id='feature_evaluation') as feature_evaluation:
        # Calculate feature importance
        calculate_importance = DockerOperator(
            task_id='calculate_feature_importance',
            image='feature_extraction',
            container_name='feature_importance',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/feature_evaluation/calculate_importance.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'EXTRACTION_RUN_ID': "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            },
        )
        
        # Generate feature metadata
        generate_metadata = DockerOperator(
            task_id='generate_feature_metadata',
            image='feature_extraction',
            container_name='feature_metadata',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/feature_evaluation/generate_metadata.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'EXTRACTION_RUN_ID': "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            },
        )
        
        # Create feature sets
        create_feature_sets = DockerOperator(
            task_id='create_feature_sets',
            image='feature_extraction',
            container_name='feature_sets',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/feature_evaluation/create_feature_sets.py',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'EXTRACTION_RUN_ID': "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
                'SELECTION_METHOD': 'mrmr',  # Minimum Redundancy Maximum Relevance
            },
        )
        
        # Set dependencies within the group
        calculate_importance >> generate_metadata >> create_feature_sets
    
    # Mark timeseries data as processed
    mark_timeseries_processed = PostgresOperator(
        task_id='mark_timeseries_processed',
        postgres_conn_id='timescaledb',
        sql="""
        -- Update Aarslev data
        UPDATE timeseries.aarslev_data
        SET _feature_extracted = true,
            _feature_extracted_at = NOW()
        WHERE _feature_extracted = false
        AND _processed_at IS NOT NULL;
        
        -- Update Knudjepsen data
        UPDATE timeseries.knudjepsen_data
        SET _feature_extracted = true,
            _feature_extracted_at = NOW()
        WHERE _feature_extracted = false
        AND _processed_at IS NOT NULL;
        """,
    )
    
    # Update extraction run status
    update_extraction_status = PythonOperator(
        task_id='update_extraction_run_status',
        python_callable=_update_extraction_run_status,
    )
    
    # Log feature statistics
    log_stats = PythonOperator(
        task_id='log_feature_stats',
        python_callable=_log_feature_stats,
    )
    
    # Notify metadata service about new features
    notify_metadata = SimpleHttpOperator(
        task_id='notify_metadata_service',
        http_conn_id='metadata_service',
        endpoint='/api/features/update',
        method='POST',
        data=json.dumps({
            "extraction_run_id": "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            "timestamp": "{{ ts }}"
        }),
        headers={"Content-Type": "application/json"},
    )
    
    # Notify the Elixir API that feature extraction is complete
    notify_api = SimpleHttpOperator(
        task_id='notify_ingestion_api',
        http_conn_id='elixir_api',
        endpoint='/api/pipeline/status',
        method='POST',
        data=json.dumps({
            "status": "features_extracted",
            "extraction_run_id": "{{ ti.xcom_pull(key='feature_extraction_run_id') }}",
            "timestamp": "{{ ts }}"
        }),
        headers={"Content-Type": "application/json"},
    )
    
    # Define the main flow
    chain(
        check_timeseries_data,
        register_extraction_run,
        [extract_statistical_features, extract_temporal_features, extract_domain_features],
        feature_evaluation,
        mark_timeseries_processed,
        update_extraction_status,
        log_stats,
        notify_metadata,
        notify_api
    ) 