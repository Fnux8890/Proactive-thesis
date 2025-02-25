defmodule IngestionService.MetricsController do
  use Phoenix.Controller, namespace: IngestionService
  
  alias IngestionService.Repo
  
  @doc """
  Returns detailed metrics about the ingestion pipeline including:
  - Throughput (files/records processed per minute)
  - Error rates
  - Processing times
  - Resource usage
  """
  def index(conn, params) do
    # Get time range from parameters with defaults
    time_range = Map.get(params, "range", "hour")
    
    metrics = case time_range do
      "hour" -> get_metrics_for_past_hour()
      "day" -> get_metrics_for_past_day()
      "week" -> get_metrics_for_past_week()
      _ -> get_metrics_for_past_hour()
    end
    
    json(conn, %{
      metrics: metrics,
      range: time_range,
      timestamp: DateTime.utc_now()
    })
  end
  
  @doc """
  Returns only the latest metrics snapshot 
  """
  def latest(conn, _params) do
    # Fetch the most recent metrics
    latest_metrics = get_latest_metrics()
    
    json(conn, latest_metrics)
  end
  
  # Get metrics for the past hour with minute granularity
  defp get_metrics_for_past_hour do
    # Query Redis time series or aggregated metrics
    # This would use Redis time series if available or query from DB
    now = DateTime.utc_now()
    
    # Get hourly processing rates from DB
    {:ok, result} = Repo.query("""
      SELECT 
        date_trunc('minute', processed_at) as minute,
        COUNT(*) as count,
        SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success_count,
        SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as error_count,
        SUM(record_count) as records_processed
      FROM ingestion_results
      WHERE processed_at > NOW() - INTERVAL '1 hour'
      GROUP BY minute
      ORDER BY minute
    """, [])
    
    # Format query results
    hourly_metrics = result.rows
    |> Enum.map(fn [minute, count, success_count, error_count, records] ->
      %{
        timestamp: minute,
        files_processed: count,
        success_count: success_count,
        error_count: error_count,
        records_processed: records
      }
    end)
    
    # Get processing time statistics
    {:ok, timing_result} = Repo.query("""
      SELECT 
        file_type,
        AVG(extract(epoch from (processed_at - lead(processed_at) OVER (PARTITION BY file_path ORDER BY processed_at DESC)))) as avg_processing_time
      FROM ingestion_results
      WHERE processed_at > NOW() - INTERVAL '1 hour'
      GROUP BY file_type
    """, [])
    
    processing_times = timing_result.rows
    |> Enum.map(fn [file_type, avg_time] ->
      {file_type, avg_time}
    end)
    |> Enum.into(%{})
    
    %{
      hourly_metrics: hourly_metrics,
      processing_times: processing_times,
      period_start: DateTime.add(now, -3600, :second),
      period_end: now
    }
  end
  
  # Get metrics for the past day with hour granularity
  defp get_metrics_for_past_day do
    # Similar to hour metrics but with different granularity
    now = DateTime.utc_now()
    
    {:ok, result} = Repo.query("""
      SELECT 
        date_trunc('hour', processed_at) as hour,
        COUNT(*) as count,
        SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success_count,
        SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as error_count,
        SUM(record_count) as records_processed
      FROM ingestion_results
      WHERE processed_at > NOW() - INTERVAL '1 day'
      GROUP BY hour
      ORDER BY hour
    """, [])
    
    daily_metrics = result.rows
    |> Enum.map(fn [hour, count, success_count, error_count, records] ->
      %{
        timestamp: hour,
        files_processed: count,
        success_count: success_count,
        error_count: error_count,
        records_processed: records
      }
    end)
    
    %{
      daily_metrics: daily_metrics,
      period_start: DateTime.add(now, -86400, :second),
      period_end: now
    }
  end
  
  # Get metrics for the past week with day granularity
  defp get_metrics_for_past_week do
    # Similar to hour metrics but with different granularity
    now = DateTime.utc_now()
    
    {:ok, result} = Repo.query("""
      SELECT 
        date_trunc('day', processed_at) as day,
        COUNT(*) as count,
        SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success_count,
        SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as error_count,
        SUM(record_count) as records_processed
      FROM ingestion_results
      WHERE processed_at > NOW() - INTERVAL '7 days'
      GROUP BY day
      ORDER BY day
    """, [])
    
    weekly_metrics = result.rows
    |> Enum.map(fn [day, count, success_count, error_count, records] ->
      %{
        timestamp: day,
        files_processed: count,
        success_count: success_count,
        error_count: error_count,
        records_processed: records
      }
    end)
    
    %{
      weekly_metrics: weekly_metrics,
      period_start: DateTime.add(now, -604800, :second),
      period_end: now
    }
  end
  
  # Get the latest metrics snapshot
  defp get_latest_metrics do
    # Get Redis counters for real-time stats
    {:ok, keys} = Redix.command(:redix, ["KEYS", "ingestion:counters:*"])
    
    counters = keys
    |> Enum.map(fn key ->
      {:ok, value} = Redix.command(:redix, ["GET", key])
      {key, String.to_integer(value)}
    end)
    |> Enum.into(%{})
    
    # Get latest process rates from DB
    {:ok, result} = Repo.query("""
      SELECT 
        COUNT(*) as total_processed,
        SUM(record_count) as total_records,
        SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success_count,
        SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as error_count
      FROM ingestion_results
      WHERE processed_at > NOW() - INTERVAL '5 minutes'
    """, [])
    
    [[total, records, success, errors]] = result.rows
    
    # Get latest processing times
    {:ok, times_result} = Repo.query("""
      SELECT 
        file_type,
        source,
        AVG(extract(epoch from (processed_at - lead(processed_at) OVER (PARTITION BY file_path ORDER BY processed_at DESC)))) as avg_processing_time
      FROM ingestion_results
      WHERE processed_at > NOW() - INTERVAL '30 minutes'
      GROUP BY file_type, source
    """, [])
    
    processing_times = times_result.rows
    |> Enum.map(fn [file_type, source, avg_time] ->
      {"#{source}_#{file_type}", avg_time}
    end)
    |> Enum.into(%{})
    
    %{
      timestamp: DateTime.utc_now(),
      counters: counters,
      last_5_minutes: %{
        total_processed: total,
        total_records: records,
        success_count: success,
        error_count: errors
      },
      processing_times: processing_times
    }
  end
end 