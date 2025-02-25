defmodule IngestionService.Pipeline.Processor do
  use GenStage
  require Logger
  
  @moduledoc """
  A GenStage consumer-producer that processes data files based on their type.
  Subscribes to the Producer and forwards processed data to the Validator.
  """
  
  # Client API
  
  def start_link(opts) do
    type = Keyword.fetch!(opts, :type)
    id = Keyword.get(opts, :id, :"processor_#{type}")
    GenStage.start_link(__MODULE__, opts, name: id)
  end
  
  # Server callbacks
  
  @impl true
  def init(opts) do
    type = Keyword.fetch!(opts, :type)
    Logger.info("Starting #{type} processor")
    
    # Configure as consumer-producer with demand dispatcher
    # Subscribe to the producer
    {:consumer_producer, %{type: type}, subscribe_to: [{IngestionService.Pipeline.Producer, selector: &selector(&1, type)}]}
  end
  
  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Processing #{length(events)} #{state.type} events")
    
    # Process each event in parallel using Flow
    processed_events = 
      Flow.from_enumerable(events)
      |> Flow.map(fn event -> process_event(event, state.type) end)
      |> Enum.to_list()
    
    # Emit processed events for the next stage
    {:noreply, processed_events, state}
  end
  
  # Process a single event based on type
  defp process_event(event, type) do
    start_time = System.monotonic_time()
    
    # Extract data from the file
    result = try do
      {data, metadata} = extract_data(event, type)
      
      # Track telemetry for this processing
      duration = System.monotonic_time() - start_time
      :telemetry.execute(
        [:ingestion_service, :pipeline, :process],
        %{duration: duration, success: true},
        %{file_type: type, path: event.file_path}
      )
      
      # Return processed event
      %{event | 
        data: data, 
        metadata: metadata, 
        status: :processed,
        processed_at: DateTime.utc_now()
      }
    rescue
      error ->
        # Log the error
        Logger.error("Error processing #{type} file #{event.file_path}: #{inspect(error)}")
        
        # Track telemetry for this failed processing
        duration = System.monotonic_time() - start_time
        :telemetry.execute(
          [:ingestion_service, :pipeline, :process],
          %{duration: duration, success: false},
          %{file_type: type, path: event.file_path, error: inspect(error)}
        )
        
        # Return event with error
        %{event | 
          status: :error, 
          error: inspect(error),
          processed_at: DateTime.utc_now()
        }
    end
    
    result
  end
  
  # Extract data from CSV files
  defp extract_data(%{file_path: path} = _event, :csv) do
    # Use NimbleCSV to parse the file
    parser = NimbleCSV.RFC4180
    
    # Read and parse the CSV file
    data = 
      path
      |> File.stream!()
      |> parser.parse_stream()
      |> Stream.with_index()
      |> Enum.reduce({[], nil}, fn {row, idx}, {rows, headers} ->
        if idx == 0 do
          # First row is headers
          {rows, row}
        else
          # Convert row to map with header keys
          row_map = Enum.zip(headers, row) |> Enum.into(%{})
          {[row_map | rows], headers}
        end
      end)
      |> elem(0)
      |> Enum.reverse()
    
    # Extract metadata from the CSV
    headers = data
      |> List.first()
      |> Map.keys()
    
    row_count = length(data)
    
    metadata = %{
      headers: headers,
      row_count: row_count,
      format: :csv
    }
    
    {data, metadata}
  end
  
  # Extract data from JSON files
  defp extract_data(%{file_path: path} = _event, :json) do
    # Parse JSON file
    data = 
      path
      |> File.read!()
      |> Jason.decode!()
    
    # Extract metadata depending on structure
    metadata = cond do
      is_list(data) ->
        %{
          record_count: length(data),
          format: :json_array
        }
      is_map(data) ->
        %{
          keys: Map.keys(data),
          format: :json_object
        }
      true ->
        %{format: :json_unknown}
    end
    
    {data, metadata}
  end
  
  # Extract data from Excel files
  defp extract_data(%{file_path: path} = _event, :excel) do
    # Use Xlsxir to parse Excel file
    {:ok, ref} = Xlsxir.multi_extract(path)
    
    # Get sheet names
    sheet_names = Xlsxir.multi_info(ref, :sheets)
    
    # Process each sheet
    data = Enum.map(sheet_names, fn sheet ->
      rows = Xlsxir.get_list(ref, sheet)
      headers = List.first(rows)
      
      # Convert each row to map with header keys
      sheet_data = 
        rows
        |> Enum.drop(1)
        |> Enum.map(fn row ->
          Enum.zip(headers, row) |> Enum.into(%{})
        end)
      
      # Return data for this sheet
      {sheet, sheet_data}
    end)
    |> Enum.into(%{})
    
    # Close the Excel file
    Xlsxir.close(ref)
    
    # Extract metadata
    sheet_counts = Enum.map(data, fn {sheet, rows} -> {sheet, length(rows)} end)
    |> Enum.into(%{})
    
    metadata = %{
      sheets: Map.keys(data),
      row_counts: sheet_counts,
      format: :excel
    }
    
    {data, metadata}
  end
  
  # Selector function to filter events based on processor type
  defp selector(event, type) do
    event.file_type == type
  end
end 