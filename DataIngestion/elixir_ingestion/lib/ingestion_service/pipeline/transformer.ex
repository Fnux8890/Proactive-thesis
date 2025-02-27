defmodule IngestionService.Pipeline.Transformer do
  use GenStage
  require Logger

  @moduledoc """
  A GenStage consumer-producer that transforms and normalizes validated data.
  Subscribes to the Validator and forwards transformed data to the Writer.
  """

  # Client API

  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  # Server callbacks

  @impl true
  def init(:ok) do
    Logger.info("Starting data transformer")

    # Configure as consumer-producer
    # Subscribe to the validator
    {:consumer_producer, %{},
     subscribe_to: [{IngestionService.Pipeline.Validator, max_demand: 10}]}
  end

  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Transforming #{length(events)} events")

    # Skip events that failed validation or earlier stages
    filtered_events =
      Enum.filter(events, fn event ->
        event.status != :error && event.status != :invalid
      end)

    # Process each event in parallel using Flow
    transformed_events =
      Flow.from_enumerable(filtered_events)
      |> Flow.map(&transform_event/1)
      |> Enum.to_list()

    # Emit transformed events for the next stage
    {:noreply, transformed_events, state}
  end

  # Transform a single event
  defp transform_event(event) do
    start_time = System.monotonic_time()

    try do
      # Apply transformations based on source and type
      transformed_data = transform_data(event)

      # Track telemetry for this transformation
      duration = System.monotonic_time() - start_time

      :telemetry.execute(
        [:ingestion_service, :pipeline, :transform],
        %{duration: duration, success: true},
        %{source: event.source, file_type: event.file_type}
      )

      # Return event with transformed data
      %{
        event
        | transformed_data: transformed_data,
          status: :transformed,
          transformed_at: DateTime.utc_now()
      }
    rescue
      error ->
        # Log the error
        Logger.error("Error transforming data from #{event.file_path}: #{inspect(error)}")

        # Track telemetry for this failed transformation
        duration = System.monotonic_time() - start_time

        :telemetry.execute(
          [:ingestion_service, :pipeline, :transform],
          %{duration: duration, success: false},
          %{source: event.source, file_type: event.file_type, error: inspect(error)}
        )

        # Return event with error
        %{event | status: :error, error: inspect(error), transformed_at: DateTime.utc_now()}
    end
  end

  # Apply transformations based on source and type
  defp transform_data(%{source: :aarslev, file_type: :csv} = event) do
    # Transform Aarslev CSV data (greenhouse measurements)
    transform_greenhouse_data(event.data)
  end

  defp transform_data(%{source: :knudjepsen, file_type: :csv} = event) do
    # Transform Knudjepsen CSV data
    transform_greenhouse_data(event.data)
  end

  defp transform_data(event) do
    # Generic transformation for other data types
    transform_generic_data(event.data, event.file_type)
  end

  # Transform greenhouse data
  defp transform_greenhouse_data(data) do
    # Transform each row to standard format
    Enum.map(data, fn row ->
      # Normalize timestamps
      start_time = normalize_timestamp(Map.get(row, "Start"))
      end_time = normalize_timestamp(Map.get(row, "End"))

      # Find and normalize measurement fields
      temperature = find_and_normalize_field(row, "temp", :float)
      humidity = find_and_normalize_field(row, ["humid", "rh"], :float)
      co2 = find_and_normalize_field(row, "co2", :float)
      flow = find_and_normalize_field(row, "flow", :float)

      # Create standardized record
      %{
        timestamp_start: start_time,
        timestamp_end: end_time,
        duration_minutes: time_difference_minutes(start_time, end_time),
        measurements: %{
          temperature: temperature,
          humidity: humidity,
          co2: co2,
          flow: flow
        },
        original: row
      }
    end)
  end

  # Transform generic data
  defp transform_generic_data(data, :json) when is_list(data) do
    # Transform JSON array
    Enum.map(data, fn item ->
      Map.put(item, :processed_at, DateTime.utc_now())
    end)
  end

  defp transform_generic_data(data, :json) when is_map(data) do
    # Transform JSON object
    Map.put(data, :processed_at, DateTime.utc_now())
  end

  defp transform_generic_data(data, :excel) when is_map(data) do
    # Transform Excel data
    Enum.map(data, fn {sheet, rows} ->
      transformed_rows =
        Enum.map(rows, fn row ->
          Map.put(row, :processed_at, DateTime.utc_now())
        end)

      {sheet, transformed_rows}
    end)
    |> Enum.into(%{})
  end

  defp transform_generic_data(data, _) do
    # Fallback transformation
    data
  end

  # Normalize timestamp to ISO8601
  defp normalize_timestamp(nil), do: nil

  defp normalize_timestamp(timestamp) when is_binary(timestamp) do
    # Try to parse the timestamp
    case DateTime.from_iso8601(timestamp) do
      {:ok, datetime, _} ->
        datetime

      _ ->
        # Try different formats
        case Timex.parse(timestamp, "{YYYY}-{0M}-{0D} {h24}:{m}") do
          {:ok, datetime} ->
            datetime

          _ ->
            case Timex.parse(timestamp, "{YYYY}-{0M}-{0D} {h24}:{m}:{s}") do
              {:ok, datetime} -> datetime
              _ -> nil
            end
        end
    end
  end

  defp normalize_timestamp(_), do: nil

  # Find a field by partial name and normalize its value
  defp find_and_normalize_field(row, field_pattern, type) when is_binary(field_pattern) do
    # Find field matching pattern
    field_name =
      Enum.find(Map.keys(row), fn key ->
        String.contains?(String.downcase(key), String.downcase(field_pattern))
      end)

    if field_name do
      normalize_value(Map.get(row, field_name), type)
    else
      nil
    end
  end

  defp find_and_normalize_field(row, field_patterns, type) when is_list(field_patterns) do
    # Try each pattern in the list
    Enum.reduce_while(field_patterns, nil, fn pattern, _acc ->
      result = find_and_normalize_field(row, pattern, type)
      if result, do: {:halt, result}, else: {:cont, nil}
    end)
  end

  # Normalize values to specified type
  defp normalize_value(value, :float) when is_binary(value) do
    case Float.parse(value) do
      {float, _} -> float
      :error -> nil
    end
  end

  defp normalize_value(value, :float) when is_number(value), do: value / 1
  defp normalize_value(value, :integer) when is_binary(value), do: String.to_integer(value)
  defp normalize_value(value, :integer) when is_number(value), do: trunc(value)
  defp normalize_value(value, :string), do: to_string(value)
  defp normalize_value(_, _), do: nil

  # Calculate difference between timestamps in minutes
  defp time_difference_minutes(nil, _), do: nil
  defp time_difference_minutes(_, nil), do: nil

  defp time_difference_minutes(start_time, end_time) do
    DateTime.diff(end_time, start_time, :second) / 60
  end
end
