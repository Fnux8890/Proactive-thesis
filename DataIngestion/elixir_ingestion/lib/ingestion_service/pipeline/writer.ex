defmodule IngestionService.Pipeline.Writer do
  use GenStage
  require Logger
  
  @moduledoc """
  A GenStage consumer that writes transformed data to TimescaleDB.
  Subscribes to the Transformer and handles the persistence of processed data.
  """
  
  # Client API
  
  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  # Server callbacks
  
  @impl true
  def init(:ok) do
    Logger.info("Starting data writer")
    
    # Configure as consumer
    # Subscribe to the transformer
    {:consumer, %{}, subscribe_to: [{IngestionService.Pipeline.Transformer, max_demand: 10}]}
  end
  
  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Writing #{length(events)} events to database")
    
    # Process each event in parallel using Flow
    Flow.from_enumerable(events)
    |> Flow.map(&write_event/1)
    |> Enum.to_list()
    
    # Consumers don't emit events, so we don't need to return any
    {:noreply, [], state}
  end
  
  # Write a single event to the database
  defp write_event(event) do
    start_time = System.monotonic_time()
    
    # Skip if there was an error in earlier stages
    if event.status == :error do
      # Log failure
      Logger.warn("Skipping write for event with error: #{event.file_path} - #{event.error}")
      
      # Persist error information for feedback loop
      save_ingestion_result(event, false)
      event
    else
      try do
        # Write data to appropriate table based on source and type
        result = write_data_to_db(event)
        
        # Track telemetry for this write operation
        duration = System.monotonic_time() - start_time
        :telemetry.execute(
          [:ingestion_service, :pipeline, :write],
          %{duration: duration, success: true},
          %{source: event.source, file_type: event.file_type}
        )
        
        # Notify the metadata catalog of successful ingestion
        notify_metadata_service(event, true)
        
        # Persist success information for feedback loop
        save_ingestion_result(event, true)
        
        # Return updated event
        %{event |
          status: :persisted,
          persisted_at: DateTime.utc_now(),
          db_result: result
        }
      rescue
        error ->
          # Log the error
          Logger.error("Error writing data from #{event.file_path} to database: #{inspect(error)}")
          
          # Track telemetry for this failed write
          duration = System.monotonic_time() - start_time
          :telemetry.execute(
            [:ingestion_service, :pipeline, :write],
            %{duration: duration, success: false},
            %{source: event.source, file_type: event.file_type, error: inspect(error)}
          )
          
          # Notify the metadata catalog of failed ingestion
          notify_metadata_service(event, false)
          
          # Persist error information for feedback loop
          save_ingestion_result(event, false, inspect(error))
          
          # Return error event
          %{event |
            status: :error,
            error: inspect(error),
            persisted_at: DateTime.utc_now()
          }
      end
    end
  end
  
  # Write data to appropriate database table
  defp write_data_to_db(%{source: :aarslev} = event) do
    # Write Aarslev data to greenhouse_measurements table
    write_greenhouse_data(event)
  end
  
  defp write_data_to_db(%{source: :knudjepsen} = event) do
    # Write Knudjepsen data to greenhouse_measurements table
    write_greenhouse_data(event)
  end
  
  defp write_data_to_db(event) do
    # Generic write for other data types
    write_generic_data(event)
  end
  
  # Write greenhouse data to TimescaleDB
  defp write_greenhouse_data(%{transformed_data: data} = event) do
    # Start a DB transaction
    repo = IngestionService.Repo
    
    # Track total rows inserted
    total_inserted = Enum.count(data, fn row ->
      # Prepare schema for timescale hypertable
      params = %{
        timestamp: row.timestamp_start,
        source: Atom.to_string(event.source),
        location: extract_location(event.file_path),
        temperature: get_in(row, [:measurements, :temperature]),
        humidity: get_in(row, [:measurements, :humidity]),
        co2: get_in(row, [:measurements, :co2]),
        flow: get_in(row, [:measurements, :flow]),
        duration_minutes: row.duration_minutes,
        original_file: event.file_path,
        ingested_at: DateTime.utc_now()
      }
      
      # Insert into greenhouse_measurements table
      {:ok, _} = repo.query(
        """
        INSERT INTO greenhouse_measurements
        (timestamp, source, location, temperature, humidity, co2, flow, duration_minutes, original_file, ingested_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (timestamp, source, location)
        DO UPDATE SET 
          temperature = $4,
          humidity = $5,
          co2 = $6,
          flow = $7,
          duration_minutes = $8,
          original_file = $9,
          updated_at = NOW()
        """,
        [
          params.timestamp,
          params.source,
          params.location,
          params.temperature,
          params.humidity,
          params.co2,
          params.flow,
          params.duration_minutes,
          params.original_file,
          params.ingested_at
        ]
      )
      
      true
    end)
    
    %{rows_inserted: total_inserted}
  end
  
  # Write generic data to appropriate table
  defp write_generic_data(%{file_type: :json} = event) do
    # Write JSON data to generic_data table
    repo = IngestionService.Repo
    
    params = %{
      timestamp: DateTime.utc_now(),
      source: Atom.to_string(event.source),
      file_path: event.file_path,
      data_json: Jason.encode!(event.transformed_data),
      file_type: Atom.to_string(event.file_type)
    }
    
    {:ok, _} = repo.query(
      """
      INSERT INTO generic_data
      (timestamp, source, file_path, data_json, file_type)
      VALUES ($1, $2, $3, $4, $5)
      ON CONFLICT (file_path)
      DO UPDATE SET 
        data_json = $4,
        updated_at = NOW()
      """,
      [
        params.timestamp,
        params.source,
        params.file_path,
        params.data_json,
        params.file_type
      ]
    )
    
    %{rows_inserted: 1}
  end
  
  defp write_generic_data(%{file_type: :excel} = event) do
    # Handle Excel data similarly to JSON
    write_generic_data(%{event | file_type: :json, transformed_data: Jason.encode!(event.transformed_data)})
  end
  
  defp write_generic_data(event) do
    # Default handler for other data types
    Logger.warn("No specific writer for #{event.file_type} data, falling back to generic storage")
    write_generic_data(%{event | file_type: :json, transformed_data: Jason.encode!(event.data)})
  end
  
  # Extract location information from file path
  defp extract_location(file_path) do
    # Try to find location in the path
    path_parts = String.split(file_path, "/")
    
    # Look for specific greenhouse identifiers
    cell_match = Enum.find(path_parts, fn part ->
      String.contains?(String.downcase(part), "celle") || 
      String.contains?(String.downcase(part), "cell") || 
      String.contains?(String.downcase(part), "house")
    end)
    
    if cell_match do
      cell_match
    else
      # If no specific identifier, use source directory
      case Enum.find_index(path_parts, &(&1 in ["aarslev", "knudjepsen"])) do
        nil -> "unknown"
        idx when idx + 1 < length(path_parts) -> Enum.at(path_parts, idx + 1)
        _ -> "unknown"
      end
    end
  end
  
  # Notify metadata service about ingestion
  defp notify_metadata_service(event, success) do
    # Prepare metadata
    metadata = %{
      file_path: event.file_path,
      file_type: event.file_type,
      source: event.source,
      processed_at: DateTime.utc_now(),
      status: if(success, do: "success", else: "error"),
      error: Map.get(event, :error),
      record_count: count_records(event.transformed_data),
      schema: extract_schema(event.transformed_data)
    }
    
    # Send to metadata service (non-blocking)
    Task.start(fn ->
      service_url = System.get_env("METADATA_SERVICE_URL") || "http://metadata-catalog:5000"
      
      request = %{
        method: :post,
        url: "#{service_url}/api/metadata",
        headers: [{"content-type", "application/json"}],
        body: Jason.encode!(metadata)
      }
      
      Finch.build(request.method, request.url, request.headers, request.body)
      |> Finch.request(IngestionService.Finch)
      |> case do
        {:ok, response} ->
          Logger.debug("Metadata service notification successful: #{inspect(response.status)}")
        {:error, error} ->
          Logger.error("Failed to notify metadata service: #{inspect(error)}")
      end
    end)
  end
  
  # Save ingestion result to database for feedback loop
  defp save_ingestion_result(event, success, error_message \\ nil) do
    repo = IngestionService.Repo
    
    params = %{
      file_path: event.file_path,
      source: Atom.to_string(event.source),
      file_type: Atom.to_string(event.file_type),
      success: success,
      processed_at: DateTime.utc_now(),
      error_message: error_message,
      record_count: count_records(event.transformed_data)
    }
    
    {:ok, _} = repo.query(
      """
      INSERT INTO ingestion_results
      (file_path, source, file_type, success, processed_at, error_message, record_count)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      """,
      [
        params.file_path,
        params.source,
        params.file_type,
        params.success,
        params.processed_at,
        params.error_message,
        params.record_count
      ]
    )
  rescue
    error ->
      Logger.error("Failed to save ingestion result: #{inspect(error)}")
  end
  
  # Count records in transformed data
  defp count_records(nil), do: 0
  
  defp count_records(data) when is_list(data) do
    length(data)
  end
  
  defp count_records(data) when is_map(data) do
    # For maps (like Excel data with multiple sheets), sum all records
    if Map.has_key?(data, :sheets) do
      # Excel-like data with sheets
      data.sheets
      |> Enum.map(fn sheet ->
        sheet_data = Map.get(data, sheet, [])
        length(sheet_data)
      end)
      |> Enum.sum()
    else
      # Just one map record
      1
    end
  end
  
  defp count_records(_), do: 0
  
  # Extract schema information from data
  defp extract_schema(nil), do: nil
  
  defp extract_schema(data) when is_list(data) and length(data) > 0 do
    # Get first record for schema
    first = List.first(data)
    extract_schema(first)
  end
  
  defp extract_schema(data) when is_map(data) do
    # Extract keys and infer types
    data
    |> Map.keys()
    |> Enum.map(fn key ->
      value = Map.get(data, key)
      {key, infer_type(value)}
    end)
    |> Enum.into(%{})
  end
  
  defp extract_schema(_), do: nil
  
  # Infer type of a value
  defp infer_type(nil), do: "null"
  defp infer_type(value) when is_binary(value), do: "string"
  defp infer_type(value) when is_integer(value), do: "integer"
  defp infer_type(value) when is_float(value), do: "float"
  defp infer_type(value) when is_boolean(value), do: "boolean"
  defp infer_type(value) when is_list(value), do: "array"
  defp infer_type(value) when is_map(value), do: "object"
  defp infer_type(_), do: "unknown"
end 