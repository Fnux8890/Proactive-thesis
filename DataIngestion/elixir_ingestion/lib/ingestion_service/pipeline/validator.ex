defmodule IngestionService.Pipeline.Validator do
  use GenStage
  require Logger
  
  @moduledoc """
  A GenStage consumer-producer that validates processed data.
  Subscribes to all Processors and forwards validated data to the Transformer.
  """
  
  # Client API
  
  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  # Server callbacks
  
  @impl true
  def init(:ok) do
    Logger.info("Starting data validator")
    
    # Configure as consumer-producer
    # Subscribe to all processor types
    {:consumer_producer, %{},
     subscribe_to: [
       {:processor_csv, max_demand: 10},
       {:processor_json, max_demand: 10},
       {:processor_excel, max_demand: 10}
     ]}
  end
  
  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Validating #{length(events)} events")
    
    # Process each event in parallel using Flow
    validated_events = 
      Flow.from_enumerable(events)
      |> Flow.map(&validate_event/1)
      |> Enum.to_list()
    
    # Emit validated events for the next stage
    {:noreply, validated_events, state}
  end
  
  # Validate a single event
  defp validate_event(event) do
    start_time = System.monotonic_time()
    
    # Skip validation if there was an error in earlier stages
    if event.status == :error do
      event
    else
      try do
        # Apply data validation rules
        {valid, validation_results} = validate_data(event)
        
        # Track telemetry for this validation
        duration = System.monotonic_time() - start_time
        :telemetry.execute(
          [:ingestion_service, :pipeline, :validate],
          %{duration: duration, valid: valid},
          %{source: event.source, file_type: event.file_type}
        )
        
        # Update event with validation results
        status = if valid, do: :validated, else: :invalid
        
        %{event |
          validation_results: validation_results,
          status: status,
          validated_at: DateTime.utc_now()
        }
      rescue
        error ->
          # Log the error
          Logger.error("Error validating #{event.file_path}: #{inspect(error)}")
          
          # Track telemetry for this failed validation
          duration = System.monotonic_time() - start_time
          :telemetry.execute(
            [:ingestion_service, :pipeline, :validate],
            %{duration: duration, valid: false},
            %{source: event.source, file_type: event.file_type, error: inspect(error)}
          )
          
          # Return event with error
          %{event |
            status: :error,
            error: inspect(error),
            validated_at: DateTime.utc_now()
          }
      end
    end
  end
  
  # Validate data based on source and type
  defp validate_data(%{source: :aarslev, file_type: :csv} = event) do
    # Apply validation rules for Aarslev CSV files
    validate_greenhouse_data(event.data)
  end
  
  defp validate_data(%{source: :knudjepsen, file_type: :csv} = event) do
    # Apply validation rules for Knudjepsen CSV files
    validate_greenhouse_data(event.data)
  end
  
  defp validate_data(event) do
    # Generic validation for other data types
    validate_generic_data(event.data, event.file_type)
  end
  
  # Validate greenhouse data
  defp validate_greenhouse_data(data) do
    # Basic validation rules for greenhouse data
    results = Enum.map(data, fn row ->
      # Check for required fields
      required_fields_present = check_required_fields(row)
      
      # Check for value ranges
      temp_in_range = check_temperature_range(row)
      humidity_in_range = check_humidity_range(row)
      co2_in_range = check_co2_range(row)
      
      # Check for timestamp format and sequence
      timestamp_valid = check_timestamp(row)
      
      # Combine all validation results
      valid = required_fields_present && temp_in_range && humidity_in_range && 
              co2_in_range && timestamp_valid
      
      %{
        valid: valid,
        row: row,
        issues: %{
          required_fields: required_fields_present,
          temperature: temp_in_range,
          humidity: humidity_in_range,
          co2: co2_in_range,
          timestamp: timestamp_valid
        }
      }
    end)
    
    # Check if all rows are valid
    all_valid = Enum.all?(results, & &1.valid)
    
    {all_valid, results}
  end
  
  # Validate generic data
  defp validate_generic_data(data, file_type) do
    # Apply basic validation based on file type
    results = case file_type do
      :json -> validate_json_data(data)
      :excel -> validate_excel_data(data)
      _ -> %{valid: true, data: data}
    end
    
    {results.valid, results}
  end
  
  # Check required fields in greenhouse data
  defp check_required_fields(row) do
    # List of required fields based on data schema
    required_fields = ~w(Start End)
    
    # Check if all required fields are present and not nil/empty
    Enum.all?(required_fields, fn field ->
      Map.has_key?(row, field) && Map.get(row, field) != nil && Map.get(row, field) != ""
    end)
  end
  
  # Check temperature values are in reasonable range
  defp check_temperature_range(row) do
    # Find temperature field - could be named in different ways
    temp_field = Enum.find(Map.keys(row), fn key ->
      String.contains?(String.downcase(key), "temp")
    end)
    
    if temp_field do
      case parse_float(Map.get(row, temp_field)) do
        {:ok, temp} -> temp >= -30.0 && temp <= 50.0
        :error -> false
      end
    else
      # No temperature field - pass validation
      true
    end
  end
  
  # Check humidity values are in range 0-100%
  defp check_humidity_range(row) do
    # Find humidity field
    humidity_field = Enum.find(Map.keys(row), fn key ->
      String.contains?(String.downcase(key), "humid") || 
      String.contains?(String.downcase(key), "rh")
    end)
    
    if humidity_field do
      case parse_float(Map.get(row, humidity_field)) do
        {:ok, humidity} -> humidity >= 0.0 && humidity <= 100.0
        :error -> false
      end
    else
      # No humidity field - pass validation
      true
    end
  end
  
  # Check CO2 values are in reasonable range
  defp check_co2_range(row) do
    # Find CO2 field
    co2_field = Enum.find(Map.keys(row), fn key ->
      String.contains?(String.downcase(key), "co2")
    end)
    
    if co2_field do
      case parse_float(Map.get(row, co2_field)) do
        {:ok, co2} -> co2 >= 0.0 && co2 <= 5000.0
        :error -> false
      end
    else
      # No CO2 field - pass validation
      true
    end
  end
  
  # Check timestamp format
  defp check_timestamp(row) do
    # Find timestamp fields
    start_field = Map.get(row, "Start")
    end_field = Map.get(row, "End")
    
    # Try to parse timestamps
    start_valid = start_field && parse_datetime(start_field) != :error
    end_valid = end_field && parse_datetime(end_field) != :error
    
    start_valid && end_valid
  end
  
  # Parse float safely
  defp parse_float(value) when is_binary(value) do
    case Float.parse(value) do
      {float, _} -> {:ok, float}
      :error -> :error
    end
  end
  
  defp parse_float(value) when is_number(value), do: {:ok, value / 1}
  defp parse_float(_), do: :error
  
  # Parse datetime safely
  defp parse_datetime(value) when is_binary(value) do
    try do
      case DateTime.from_iso8601(value) do
        {:ok, datetime, _} -> {:ok, datetime}
        _ -> :error
      end
    rescue
      _ -> :error
    end
  end
  
  defp parse_datetime(_), do: :error
  
  # Validate JSON data
  defp validate_json_data(data) when is_list(data) do
    # Validate each item in the list
    results = Enum.map(data, fn item ->
      %{valid: true, data: item}
    end)
    
    %{valid: true, data: data, results: results}
  end
  
  defp validate_json_data(data) when is_map(data) do
    # Just a basic validation that it's a map
    %{valid: true, data: data}
  end
  
  defp validate_json_data(_), do: %{valid: false, reason: "Invalid JSON data structure"}
  
  # Validate Excel data
  defp validate_excel_data(data) when is_map(data) do
    # Validate each sheet
    results = Enum.map(data, fn {sheet, rows} ->
      {sheet, %{valid: true, data: rows}}
    end)
    |> Enum.into(%{})
    
    %{valid: true, data: data, results: results}
  end
  
  defp validate_excel_data(_), do: %{valid: false, reason: "Invalid Excel data structure"}
end 