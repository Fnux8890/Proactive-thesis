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
    id = Keyword.fetch!(opts, :id)
    GenStage.start_link(__MODULE__, opts, name: id)
  end

  # Server callbacks

  @impl true
  def init(opts) do
    type = Keyword.fetch!(opts, :type)
    Logger.info("Starting #{type} processor")

    # Configure as consumer-producer with demand dispatcher
    # Subscribe to the producer
    {:consumer_producer, %{type: type},
     subscribe_to: [{IngestionService.Pipeline.Producer, selector: &selector(&1, type)}]}
  end

  @impl true
  def child_spec(opts) do
    id = Keyword.get(opts, :id, __MODULE__)

    %{
      id: id,
      start: {__MODULE__, :start_link, [opts]},
      type: :worker,
      restart: :permanent,
      shutdown: 500
    }
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
    result =
      try do
        {data, metadata} = extract_data(event, type)

        # Track telemetry for this processing
        duration = System.monotonic_time() - start_time

        :telemetry.execute(
          [:ingestion_service, :pipeline, :process],
          %{duration: duration, success: true},
          %{file_type: type, path: event.file_path}
        )

        # Return processed event
        %{
          event
          | data: data,
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
          %{event | status: :error, error: inspect(error), processed_at: DateTime.utc_now()}
      end

    result
  end

  # Extract data from CSV files
  defp extract_data(%{file_path: path} = _event, :csv) do
    # First, try to detect the delimiter by analyzing the file
    {delimiter, encoding} = detect_csv_format(path)
    
    # Log detected format
    Logger.info("Processing CSV file #{path} with delimiter: '#{delimiter}' and encoding: #{encoding}")
    
    # Use NimbleCSV with detected delimiter
    parser = IngestionService.Utils.CsvSeparators.get_parser(delimiter)

    # Read and parse the CSV file
    data =
      path
      |> File.stream!([], :line, encoding)
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
    headers =
      data
      |> List.first()
      |> Map.keys()

    row_count = length(data)

    metadata = %{
      headers: headers,
      row_count: row_count,
      format: :csv,
      delimiter: delimiter,
      encoding: encoding
    }

    {data, metadata}
  end
  
  # Detect CSV delimiter and encoding by analyzing file content
  defp detect_csv_format(path) do
    # Possible delimiters to check
    delimiters = [",", ";", "\t", "|"]
    
    # Try to read file with UTF-8 encoding first
    case File.read(path) do
      {:ok, content} -> 
        # File could be read with UTF-8
        delimiter = detect_delimiter(content, delimiters)
        {delimiter, :utf8}
        
      {:error, _} ->
        # Try with Latin-1 encoding
        case :file.read_file(path) do
          {:ok, content} ->
            content_latin1 = :unicode.characters_to_binary(content, :latin1, :utf8)
            delimiter = detect_delimiter(content_latin1, delimiters)
            {delimiter, :latin1}
            
          {:error, _} ->
            # Default fallback if nothing works
            {",", :utf8}
        end
    end
  end
  
  # Detect the most likely delimiter by analyzing file content
  defp detect_delimiter(content, delimiters) do
    # Take first few lines for analysis
    first_lines = 
      content
      |> String.split("\n", parts: 6)
      |> Enum.take(5)
    
    # Count occurrences of each delimiter in each line
    counts = Enum.map(delimiters, fn delimiter ->
      count = Enum.reduce(first_lines, 0, fn line, acc ->
        acc + (String.split(line, delimiter) |> length) - 1
      end)
      {delimiter, count}
    end)
    
    # Select the delimiter with the highest count
    {best_delimiter, _} = 
      counts
      |> Enum.max_by(fn {_, count} -> count end, fn -> {",", 0} end)
    
    best_delimiter
  end

  # Extract data from JSON files
  defp extract_data(%{file_path: path} = _event, :json) do
    # Parse JSON file
    case File.read!(path) |> Jason.decode() do
      {:ok, data} ->
        # Process data based on structure type
        {processed_data, structure_type} = process_json_structure(data)
        
        # Extract metadata depending on structure
        metadata = case structure_type do
          :array ->
            %{
              record_count: length(processed_data),
              format: :json_array,
              structure: structure_type
            }

          :object ->
            %{
              keys: Map.keys(processed_data),
              format: :json_object,
              structure: structure_type
            }

          :nested_array ->
            %{
              record_count: length(processed_data),
              format: :json_nested_array,
              structure: structure_type,
              flattened: true
            }

          _ ->
            %{format: :json_unknown}
        end

        {processed_data, metadata}
        
      {:error, error} ->
        # Log the JSON parsing error
        Logger.error("Failed to parse JSON file #{path}: #{inspect(error)}")
        raise "JSON parsing error: #{inspect(error)}"
    end
  end
  
  # Process JSON data based on its structure
  defp process_json_structure(data) when is_list(data) do
    # It's already an array of items
    if Enum.all?(data, &is_map/1) do
      # Array of objects - ideal format
      {data, :array}
    else
      # Array of non-objects, convert each item to a map
      {Enum.map(data, fn item -> %{"value" => item} end), :array}
    end
  end
  
  defp process_json_structure(data) when is_map(data) do
    cond do
      # Look for common array properties
      Map.has_key?(data, "data") and is_list(data["data"]) ->
        # Extract and use the "data" array
        Logger.info("Found 'data' array in JSON object, extracting...")
        {data["data"], :nested_array}
        
      Map.has_key?(data, "results") and is_list(data["results"]) ->
        # Extract and use the "results" array
        Logger.info("Found 'results' array in JSON object, extracting...")
        {data["results"], :nested_array}
        
      Map.has_key?(data, "items") and is_list(data["items"]) ->
        # Extract and use the "items" array
        Logger.info("Found 'items' array in JSON object, extracting...")
        {data["items"], :nested_array}
        
      # Check if it's a single object with simple values
      Enum.all?(data, fn {_k, v} -> not is_map(v) and not is_list(v) end) ->
        # Single object, convert to a list with one item
        Logger.info("Processing single JSON object")
        {[data], :object}
        
      # Complex nested structure
      true ->
        # Flatten the structure by extracting all leaf key-value pairs
        Logger.info("Flattening complex nested JSON structure")
        flattened = flatten_json_structure(data)
        {[flattened], :object}
    end
  end
  
  # Fallback for any other JSON type
  defp process_json_structure(data) do
    Logger.warn("Unexpected JSON structure type: #{inspect(data)}")
    {[%{"value" => inspect(data)}], :unknown}
  end
  
  # Recursively flatten a nested JSON structure
  defp flatten_json_structure(data, prefix \\ "") do
    Enum.reduce(data, %{}, fn {key, value}, acc ->
      full_key = if prefix == "", do: key, else: "#{prefix}.#{key}"
      
      case value do
        %{} ->
          # Nested object, recursively flatten
          Map.merge(acc, flatten_json_structure(value, full_key))
          
        list when is_list(list) ->
          # For arrays, store as JSON string to preserve the data
          Map.put(acc, full_key, Jason.encode!(list))
          
        _ ->
          # Primitive value
          Map.put(acc, full_key, value)
      end
    end)
  end

  # Extract data from Excel files
  defp extract_data(%{file_path: path} = _event, :excel) do
    # Use Xlsxir to parse Excel file
    {:ok, ref} = Xlsxir.multi_extract(path)

    # Get sheet names
    sheet_names = Xlsxir.multi_info(ref, :sheets)

    # Process each sheet
    data =
      Enum.map(sheet_names, fn sheet ->
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
    sheet_counts =
      Enum.map(data, fn {sheet, rows} -> {sheet, length(rows)} end)
      |> Enum.into(%{})

    metadata = %{
      sheets: Map.keys(data),
      row_counts: sheet_counts,
      format: :excel
    }

    {data, metadata}
  end

  # Select events based on file type
  defp selector(event, type) do
    # Convert atom to string for comparison if needed
    event_type = if is_atom(event.file_type), do: event.file_type, else: String.to_atom(event.file_type)
    type_atom = if is_atom(type), do: type, else: String.to_atom(type)
    
    # Compare the types
    event_type == type_atom
  end
end
