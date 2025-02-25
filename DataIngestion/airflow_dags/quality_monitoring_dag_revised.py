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

def _check_data_quality_status(**context):
    """Check if there is new data that needs quality monitoring"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    # Check for new data in timeseries tables that hasn't been quality checked
    result = pg_hook.get_records("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(*) FILTER (WHERE _data_quality_score IS NULL) as unscored_records
        FROM (
            SELECT _data_quality_score FROM timeseries.aarslev_data 
            WHERE _processed_at >= NOW() - INTERVAL '1 day'
            UNION ALL
            SELECT _data_quality_score FROM timeseries.knudjepsen_data 
            WHERE _processed_at >= NOW() - INTERVAL '1 day'
        ) as combined
    """)
    
    if not result or result[0][0] == 0:
        print("No recent data found for quality monitoring")
        return False
    
    total_records = result[0][0]
    unscored_records = result[0][1]
    
    if unscored_records == 0:
        print(f"All {total_records} recent records already have quality scores")
        return False
    
    # Generate a unique run ID for this quality monitoring run
    run_id = f"quality_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    context['ti'].xcom_push(key='quality_monitoring_run_id', value=run_id)
    
    # Store counts in XCom for downstream tasks
    context['ti'].xcom_push(key='total_records', value=total_records)
    context['ti'].xcom_push(key='unscored_records', value=unscored_records)
    
    print(f"Found {unscored_records} out of {total_records} recent records without quality scores")
    print(f"Quality monitoring run ID: {run_id}")
    return True

def _register_monitoring_run(**context):
    """Register a new quality monitoring run in the database"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    run_id = context['ti'].xcom_pull(key='quality_monitoring_run_id')
    total_records = context['ti'].xcom_pull(key='total_records')
    unscored_records = context['ti'].xcom_pull(key='unscored_records')
    
    # Create a table for quality monitoring runs if it doesn't exist
    pg_hook.run("""
        CREATE TABLE IF NOT EXISTS metadata.quality_monitoring_runs (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            end_time TIMESTAMP WITH TIME ZONE,
            status TEXT DEFAULT 'running',
            total_records INTEGER,
            processed_records INTEGER,
            anomalies_detected INTEGER,
            configuration JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)
    
    # Insert the monitoring run record
    pg_hook.run("""
        INSERT INTO metadata.quality_monitoring_runs (
            run_id,
            total_records,
            processed_records,
            configuration
        )
        VALUES (%s, %s, 0, %s)
    """, (
        run_id,
        total_records,
        json.dumps({
            "execution_date": context['execution_date'].isoformat(),
            "dag_id": context['dag'].dag_id,
            "dag_run_id": context['dag_run'].run_id,
            "unscored_records": unscored_records
        })
    ))
    
    print(f"Registered quality monitoring run with ID: {run_id}")
    return True

def _update_monitoring_run_status(**context):
    """Update the status of the quality monitoring run"""
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    
    run_id = context['ti'].xcom_pull(key='quality_monitoring_run_id')
    
    # Get anomaly counts
    result = pg_hook.get_records("""
        SELECT COUNT(*) 
        FROM (
            SELECT _outlier_score FROM timeseries.aarslev_data 
            WHERE _outlier_score > 0.8
            AND _processed_at >= NOW() - INTERVAL '1 day'
            UNION ALL
            SELECT _outlier_score FROM timeseries.knudjepsen_data 
            WHERE _outlier_score > 0.8
            AND _processed_at >= NOW() - INTERVAL '1 day'
        ) as anomalies
    """)
    
    anomalies_detected = result[0][0] if result else 0
    
    # Get processed record count
    result = pg_hook.get_records("""
        SELECT COUNT(*) 
        FROM (
            SELECT _data_quality_score FROM timeseries.aarslev_data 
            WHERE _data_quality_score IS NOT NULL
            AND _processed_at >= NOW() - INTERVAL '1 day'
            UNION ALL
            SELECT _data_quality_score FROM timeseries.knudjepsen_data 
            WHERE _data_quality_score IS NOT NULL
            AND _processed_at >= NOW() - INTERVAL '1 day'
        ) as processed
    """)
    
    processed_records = result[0][0] if result else 0
    
    # Update the monitoring run record
    pg_hook.run("""
        UPDATE metadata.quality_monitoring_runs
        SET 
            status = 'completed',
            end_time = NOW(),
            processed_records = %s,
            anomalies_detected = %s
        WHERE run_id = %s
    """, (processed_records, anomalies_detected, run_id))
    
    print(f"Updated quality monitoring run {run_id} with {processed_records} processed records and {anomalies_detected} anomalies")
    
    # Store anomaly count for downstream tasks
    context['ti'].xcom_push(key='anomalies_detected', value=anomalies_detected)
    
    return True

def _send_anomaly_alerts(**context):
    """Send alerts if anomalies are detected"""
    anomalies_detected = context['ti'].xcom_pull(key='anomalies_detected') or 0
    
    if anomalies_detected == 0:
        print("No anomalies detected, no alerts needed")
        return False
    
    # In a real system, this would send emails, Slack messages, etc.
    print(f"ALERT: {anomalies_detected} anomalies detected in recent data!")
    
    # For demonstration purposes, we'll just log the alert
    pg_hook = PostgresHook(postgres_conn_id='timescaledb')
    run_id = context['ti'].xcom_pull(key='quality_monitoring_run_id')
    
    # Create alerts table if it doesn't exist
    pg_hook.run("""
        CREATE TABLE IF NOT EXISTS metadata.quality_alerts (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL,
            alert_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            alert_type TEXT NOT NULL,
            alert_message TEXT NOT NULL,
            severity TEXT NOT NULL,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_at TIMESTAMP WITH TIME ZONE,
            acknowledged_by TEXT
        )
    """)
    
    # Log the alert
    pg_hook.run("""
        INSERT INTO metadata.quality_alerts (
            run_id,
            alert_type,
            alert_message,
            severity
        )
        VALUES (%s, %s, %s, %s)
    """, (
        run_id,
        'anomaly_detection',
        f"{anomalies_detected} anomalies detected in recent data",
        'high' if anomalies_detected > 10 else 'medium'
    ))
    
    return True

with DAG(
    'quality_monitoring_pipeline_revised',
    default_args=default_args,
    description='Revised quality monitoring pipeline for data quality assessment',
    schedule_interval=timedelta(hours=12),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['quality', 'monitoring', 'revised'],
) as dag:
    
    # Check if there's data that needs quality monitoring
    check_quality_status = PythonOperator(
        task_id='check_data_quality_status',
        python_callable=_check_data_quality_status,
    )
    
    # Register the quality monitoring run
    register_monitoring_run = PythonOperator(
        task_id='register_monitoring_run',
        python_callable=_register_monitoring_run,
    )
    
    # Data Quality Assessment Group
    with TaskGroup(group_id='data_quality_assessment') as data_quality_assessment:
        # Run data quality checks on Aarslev data
        check_aarslev_quality = DockerOperator(
            task_id='check_aarslev_quality',
            image='quality_monitor',
            container_name='aarslev_quality',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/quality_checks/assess_quality.py --source aarslev',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'MONITORING_RUN_ID': "{{ ti.xcom_pull(key='quality_monitoring_run_id') }}",
            },
        )
        
        # Run data quality checks on Knudjepsen data
        check_knudjepsen_quality = DockerOperator(
            task_id='check_knudjepsen_quality',
            image='quality_monitor',
            container_name='knudjepsen_quality',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/quality_checks/assess_quality.py --source knudjepsen',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'MONITORING_RUN_ID': "{{ ti.xcom_pull(key='quality_monitoring_run_id') }}",
            },
        )
    
    # Anomaly Detection Group
    with TaskGroup(group_id='anomaly_detection') as anomaly_detection:
        # Detect anomalies in Aarslev data
        detect_aarslev_anomalies = DockerOperator(
            task_id='detect_aarslev_anomalies',
            image='quality_monitor',
            container_name='aarslev_anomalies',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/anomaly_detection/detect_anomalies.py --source aarslev',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'MONITORING_RUN_ID': "{{ ti.xcom_pull(key='quality_monitoring_run_id') }}",
            },
        )
        
        # Detect anomalies in Knudjepsen data
        detect_knudjepsen_anomalies = DockerOperator(
            task_id='detect_knudjepsen_anomalies',
            image='quality_monitor',
            container_name='knudjepsen_anomalies',
            api_version='auto',
            auto_remove=True,
            docker_url='unix://var/run/docker.sock',
            network_mode='data-ingestion_network',
            command='uv run /app/anomaly_detection/detect_anomalies.py --source knudjepsen',
            environment={
                'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
                'MONITORING_RUN_ID': "{{ ti.xcom_pull(key='quality_monitoring_run_id') }}",
            },
        )
    
    # Update monitoring run status
    update_monitoring_status = PythonOperator(
        task_id='update_monitoring_run_status',
        python_callable=_update_monitoring_run_status,
    )
    
    # Send alerts if anomalies are detected
    send_alerts = PythonOperator(
        task_id='send_anomaly_alerts',
        python_callable=_send_anomaly_alerts,
    )
    
    # Generate quality report
    generate_report = DockerOperator(
        task_id='generate_quality_report',
        image='quality_monitor',
        container_name='quality_report',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='data-ingestion_network',
        command='uv run /app/reporting/generate_report.py',
        environment={
            'PROCESSING_DB_URL': '{{ conn.timescaledb.get_uri() }}',
            'MONITORING_RUN_ID': "{{ ti.xcom_pull(key='quality_monitoring_run_id') }}",
        },
    )
    
    # Notify metadata service about quality assessment
    notify_metadata = SimpleHttpOperator(
        task_id='notify_metadata_service',
        http_conn_id='metadata_service',
        endpoint='/api/quality/update',
        method='POST',
        data=json.dumps({
            "monitoring_run_id": "{{ ti.xcom_pull(key='quality_monitoring_run_id') }}",
            "anomalies_detected": "{{ ti.xcom_pull(key='anomalies_detected') }}",
            "timestamp": "{{ ts }}"
        }),
        headers={"Content-Type": "application/json"},
    )
    
    # Define the main flow
    chain(
        check_quality_status,
        register_monitoring_run,
        data_quality_assessment,
        anomaly_detection,
        update_monitoring_status,
        send_alerts,
        generate_report,
        notify_metadata
    ) 