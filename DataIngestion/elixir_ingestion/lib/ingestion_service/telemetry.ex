defmodule IngestionService.Telemetry do
  use Supervisor
  import Telemetry.Metrics

  def start_link(arg) do
    Supervisor.start_link(__MODULE__, arg, name: __MODULE__)
  end

  @impl true
  def init(_arg) do
    children = [
      # Telemetry poller will execute the given period measurements
      # every 10_000ms. Learn more here: https://hexdocs.pm/telemetry_metrics
      {:telemetry_poller, measurements: periodic_measurements(), period: 10_000}
      # Add reporters as children of your supervision tree.
      # {Telemetry.Metrics.ConsoleReporter, metrics: metrics()}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end

  def metrics do
    [
      # Phoenix Metrics
      summary("phoenix.endpoint.start.system_time",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.endpoint.stop.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.start.system_time",
        tags: [:route],
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.exception.duration",
        tags: [:route],
        unit: {:native, :millisecond}
      ),
      summary("phoenix.router_dispatch.stop.duration",
        tags: [:route],
        unit: {:native, :millisecond}
      ),
      summary("phoenix.socket_connected.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.channel_joined.duration",
        unit: {:native, :millisecond}
      ),
      summary("phoenix.channel_handled_in.duration",
        tags: [:event],
        unit: {:native, :millisecond}
      ),

      # Database Metrics
      summary("ingestion_service.repo.query.total_time",
        unit: {:native, :millisecond},
        description: "The sum of the other measurements"
      ),
      summary("ingestion_service.repo.query.decode_time",
        unit: {:native, :millisecond},
        description: "The time spent decoding the data received from the database"
      ),
      summary("ingestion_service.repo.query.query_time",
        unit: {:native, :millisecond},
        description: "The time spent executing the query"
      ),
      summary("ingestion_service.repo.query.queue_time",
        unit: {:native, :millisecond},
        description: "The time spent waiting for a database connection"
      ),
      summary("ingestion_service.repo.query.idle_time",
        unit: {:native, :millisecond},
        description:
          "The time the connection spent waiting before being checked out for the query"
      ),

      # Pipeline Metrics
      counter("ingestion_service.pipeline.process.count", 
        tags: [:file_type],
        description: "The number of files processed"
      ),
      summary("ingestion_service.pipeline.process.duration",
        tags: [:file_type, :success],
        unit: {:native, :millisecond},
        description: "The time spent processing files"
      ),
      counter("ingestion_service.pipeline.validate.count",
        tags: [:source, :valid],
        description: "The number of validations performed"
      ),
      summary("ingestion_service.pipeline.validate.duration",
        tags: [:source, :valid],
        unit: {:native, :millisecond},
        description: "The time spent validating data"
      ),
      counter("ingestion_service.pipeline.transform.count",
        tags: [:source],
        description: "The number of transformations performed"
      ),
      summary("ingestion_service.pipeline.transform.duration",
        tags: [:source, :success],
        unit: {:native, :millisecond},
        description: "The time spent transforming data"
      ),
      counter("ingestion_service.pipeline.write.count",
        tags: [:source],
        description: "The number of database writes performed"
      ),
      summary("ingestion_service.pipeline.write.duration",
        tags: [:source, :success],
        unit: {:native, :millisecond},
        description: "The time spent writing data to the database"
      ),

      # VM Metrics
      summary("vm.memory.total", unit: {:byte, :kilobyte}),
      summary("vm.total_run_queue_lengths.total"),
      summary("vm.total_run_queue_lengths.cpu"),
      summary("vm.total_run_queue_lengths.io")
    ]
  end

  defp periodic_measurements do
    [
      # A module, function and arguments to be invoked periodically.
      # This function must call :telemetry.execute/3 and a metric must be added above.
      # {IngestionService, :count_users, []}
    ]
  end
  
  # Handler for telemetry events
  def handle_event([:ingestion_service, :pipeline, event_name], measurements, metadata, _config) do
    # Log the telemetry event
    Logger.debug("Telemetry event: #{event_name}, measurements: #{inspect(measurements)}, metadata: #{inspect(metadata)}")
    
    # Store the event in Redis for real-time metrics
    Task.start(fn ->
      {:ok, conn} = Redix.start_link(host: System.get_env("REDIS_HOST", "redis"), port: String.to_integer(System.get_env("REDIS_PORT", "6379")))
      
      # Store the event with a TTL
      event_data = Jason.encode!(%{
        timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
        event: event_name,
        measurements: measurements,
        metadata: metadata
      })
      
      # Use a list for event history
      Redix.command(conn, ["LPUSH", "ingestion:events:#{event_name}", event_data])
      Redix.command(conn, ["LTRIM", "ingestion:events:#{event_name}", 0, 999])
      
      # Update counters
      case event_name do
        :process ->
          file_type = Map.get(metadata, :file_type)
          Redix.command(conn, ["INCR", "ingestion:counters:process:#{file_type}"])
        :validate ->
          source = Map.get(metadata, :source)
          valid = Map.get(measurements, :valid, false)
          key = if valid, do: "valid", else: "invalid"
          Redix.command(conn, ["INCR", "ingestion:counters:validate:#{source}:#{key}"])
        :write ->
          source = Map.get(metadata, :source)
          success = Map.get(measurements, :success, false)
          key = if success, do: "success", else: "error"
          Redix.command(conn, ["INCR", "ingestion:counters:write:#{source}:#{key}"])
        _ -> :ok
      end
    end)
  end
end 