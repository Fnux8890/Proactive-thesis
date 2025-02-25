defmodule IngestionService.StatusController do
  use Phoenix.Controller, namespace: IngestionService
  
  alias IngestionService.Repo
  
  @doc """
  Returns the current status of the ingestion service including:
  - System information (uptime, memory usage)
  - Database connection status
  - Redis connection status
  - Pipeline statistics (files processed, success/error rates)
  - Queue information
  """
  def status(conn, _params) do
    # Get system information
    memory = :erlang.memory()
    system_info = %{
      memory: %{
        total: memory[:total],
        processes: memory[:processes],
        atom: memory[:atom],
        binary: memory[:binary]
      },
      uptime: :erlang.statistics(:wall_clock) |> elem(0),
      process_count: :erlang.system_info(:process_count)
    }
    
    # Check database connection
    db_status = case Repo.query("SELECT 1 as alive", []) do
      {:ok, _} -> :connected
      {:error, error} -> %{error: inspect(error)}
    end
    
    # Check Redis connection
    redis_status = case Redix.command(:redix, ["PING"]) do
      {:ok, "PONG"} -> :connected
      {:error, error} -> %{error: inspect(error)}
    end
    
    # Get pipeline statistics from Redis
    pipeline_stats = get_pipeline_stats()
    
    json(conn, %{
      status: :ok,
      version: Application.spec(:ingestion_service, :vsn) || "dev",
      started_at: Application.get_env(:ingestion_service, :started_at),
      system: system_info,
      connections: %{
        database: db_status,
        redis: redis_status
      },
      pipeline: pipeline_stats
    })
  end
  
  @doc """
  Returns health check status for container orchestration
  """
  def health(conn, _params) do
    # Basic health check - make sure DB and Redis are connected
    db_healthy = case Repo.query("SELECT 1 as alive", [], timeout: 1000) do
      {:ok, _} -> true
      {:error, _} -> false
    end
    
    redis_healthy = case Redix.command(:redix, ["PING"], timeout: 1000) do
      {:ok, "PONG"} -> true
      {:error, _} -> false
    end
    
    is_healthy = db_healthy and redis_healthy
    
    status_code = if is_healthy, do: 200, else: 503
    
    conn
    |> put_status(status_code)
    |> json(%{
      status: if(is_healthy, do: :ok, else: :error),
      database: db_healthy,
      redis: redis_healthy
    })
  end
  
  # Fetch pipeline statistics from Redis
  defp get_pipeline_stats do
    case Redix.command(:redix, ["KEYS", "ingestion:counters:*"]) do
      {:ok, keys} ->
        stats = keys
        |> Enum.map(fn key ->
          {:ok, value} = Redix.command(:redix, ["GET", key])
          # Parse the key to get metric type
          parts = String.split(key, ":")
          type = Enum.at(parts, 2)
          source = if length(parts) > 3, do: Enum.at(parts, 3), else: nil
          status = if length(parts) > 4, do: Enum.at(parts, 4), else: nil
          
          {type, source, status, String.to_integer(value)}
        end)
        |> Enum.reduce(%{}, fn {type, source, status, count}, acc ->
          # Build nested map from pipeline stats
          type_map = Map.get(acc, type, %{})
          
          source_map = if source do
            Map.get(type_map, source, %{})
          else
            count
          end
          
          updated_source_map = if status do
            Map.put(source_map, status, count)
          else
            source_map
          end
          
          updated_type_map = if source do
            Map.put(type_map, source, updated_source_map) 
          else 
            type_map
          end
          
          Map.put(acc, type, updated_type_map)
        end)
        
        # Add recent events
        {:ok, recent_process} = Redix.command(:redix, ["LRANGE", "ingestion:events:process", 0, 9])
        {:ok, recent_write} = Redix.command(:redix, ["LRANGE", "ingestion:events:write", 0, 9])
        
        Map.merge(stats, %{
          recent_processed: Enum.map(recent_process, &Jason.decode!/1),
          recent_writes: Enum.map(recent_write, &Jason.decode!/1)
        })
        
      {:error, _} ->
        %{error: "Failed to fetch pipeline statistics"}
    end
  end
end 