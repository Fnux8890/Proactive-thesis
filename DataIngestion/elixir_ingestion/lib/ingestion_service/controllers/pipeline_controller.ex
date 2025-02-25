defmodule IngestionService.PipelineController do
  use Phoenix.Controller, namespace: IngestionService
  
  require Logger
  
  @doc """
  Get the current status of the ingestion pipeline including:
  - Pipeline stages and their status
  - Currently processing files
  - Queue stats
  """
  def status(conn, _params) do
    # Collect status information from all pipeline stages
    pipeline_status = %{
      status: :running,
      stages: %{
        file_watcher: get_process_status(IngestionService.Pipeline.FileWatcher),
        producer: get_process_status(IngestionService.Pipeline.Producer),
        processor_csv: get_process_status(:processor_csv),
        processor_json: get_process_status(:processor_json),
        processor_excel: get_process_status(:processor_excel),
        validator: get_process_status(IngestionService.Pipeline.Validator),
        transformer: get_process_status(IngestionService.Pipeline.Transformer),
        writer: get_process_status(IngestionService.Pipeline.Writer)
      }
    }
    
    # Get currently processing files from Redis
    {:ok, processing_files} = Redix.command(:redix, ["KEYS", "ingestion:tracking:*"])
    
    processing_count = length(processing_files)
    
    # Get details about files in process (limited to 10)
    recent_files = 
      if processing_count > 0 do
        # Get 10 most recent files
        Enum.take(processing_files, 10)
        |> Enum.map(fn key ->
          {:ok, data} = Redix.command(:redix, ["HGETALL", key])
          
          # Convert to map
          data
          |> Enum.chunk_every(2)
          |> Enum.map(fn [k, v] -> {k, v} end)
          |> Enum.into(%{})
        end)
      else
        []
      end
    
    # Get queue backlog information
    queue_info = %{
      total_queued: processing_count,
      recent_files: recent_files
    }
    
    # Add supervisor information
    {:ok, supervisor_info} = get_supervisor_info()
    
    json(conn, %{
      pipeline: pipeline_status,
      queue: queue_info,
      supervisor: supervisor_info,
      timestamp: DateTime.utc_now()
    })
  end
  
  @doc """
  Reload the pipeline configuration and restart pipeline components.
  """
  def reload(conn, _params) do
    Logger.info("Reloading pipeline configuration")
    
    # Restart the pipeline supervisor
    # This is a soft restart that will maintain queue state
    :ok = Supervisor.terminate_child(IngestionService.Supervisor, IngestionService.Pipeline.Supervisor)
    {:ok, _} = Supervisor.restart_child(IngestionService.Supervisor, IngestionService.Pipeline.Supervisor)
    
    # Get updated status
    {:ok, supervisor_info} = get_supervisor_info()
    
    json(conn, %{
      status: :ok,
      message: "Pipeline configuration reloaded",
      timestamp: DateTime.utc_now(),
      supervisor: supervisor_info
    })
  end
  
  @doc """
  Purge all pending files from the pipeline queues.
  """
  def purge(conn, _params) do
    Logger.warn("Purging all pending files from pipeline queues")
    
    # Get all tracking keys
    {:ok, tracking_keys} = Redix.command(:redix, ["KEYS", "ingestion:tracking:*"])
    
    # Count purged entries
    purged_count = length(tracking_keys)
    
    # Delete all tracking entries
    if purged_count > 0 do
      Redix.command(:redix, ["DEL" | tracking_keys])
    end
    
    json(conn, %{
      status: :ok,
      message: "Pipeline queues purged",
      purged_count: purged_count,
      timestamp: DateTime.utc_now()
    })
  end
  
  @doc """
  Update pipeline configuration
  """
  def configure(conn, %{"config" => config}) do
    Logger.info("Updating pipeline configuration: #{inspect(config)}")
    
    # Store configuration in Redis
    Enum.each(config, fn {key, value} ->
      Redix.command(:redix, ["SET", "ingestion:config:#{key}", to_string(value)])
    end)
    
    # Signal components to reload config
    Phoenix.PubSub.broadcast(
      IngestionService.PubSub,
      "pipeline:control",
      {:reload_config, config}
    )
    
    json(conn, %{
      status: :ok,
      message: "Pipeline configuration updated",
      config: config,
      timestamp: DateTime.utc_now()
    })
  end
  
  # Helper to get process status
  defp get_process_status(process_name) do
    pid = Process.whereis(process_name)
    
    if pid && Process.alive?(pid) do
      # Process is running
      # Get process info
      info = Process.info(pid, [:message_queue_len, :memory, :status])
      
      %{
        status: :running,
        pid: pid,
        message_queue_len: info[:message_queue_len],
        memory: info[:memory],
        process_status: info[:status]
      }
    else
      # Process is not running
      %{
        status: :not_running
      }
    end
  end
  
  # Get supervisor info
  defp get_supervisor_info do
    supervisor_pid = Process.whereis(IngestionService.Pipeline.Supervisor)
    
    if supervisor_pid && Process.alive?(supervisor_pid) do
      # Get all children
      children = Supervisor.which_children(IngestionService.Pipeline.Supervisor)
      
      # Format children info
      children_info = children
      |> Enum.map(fn {id, pid, type, modules} ->
        %{
          id: id,
          pid: if(pid, do: inspect(pid), else: nil),
          type: type,
          modules: modules,
          status: if(pid && Process.alive?(pid), do: :running, else: :not_running)
        }
      end)
      
      {:ok, %{
        status: :running,
        pid: inspect(supervisor_pid),
        children: children_info
      }}
    else
      {:error, :not_running}
    end
  end
end 