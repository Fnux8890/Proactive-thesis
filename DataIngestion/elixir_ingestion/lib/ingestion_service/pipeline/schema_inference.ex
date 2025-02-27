defmodule IngestionService.Pipeline.SchemaInference do
  use GenStage
  require Logger

  @moduledoc """
  A GenStage consumer-producer that infers schema from data samples and
  tracks schema evolution over time. This enables the system to adapt to
  new data sources without requiring predefined schemas.
  """

  # Client API

  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @doc """
  Infers a schema from a data sample and reconciles it with an existing schema if provided.
  """
  def infer_schema(data_sample, existing_schema \\ nil) do
    inferred_schema = do_inference(data_sample)

    case existing_schema do
      nil -> inferred_schema
      existing -> merge_schemas(existing, inferred_schema)
    end
  end

  @doc """
  Retrieves the current schema for a given dataset id.
  """
  def get_schema(dataset_id) do
    GenServer.call(__MODULE__, {:get_schema, dataset_id})
  end

  @doc """
  Updates the schema for a given dataset id.
  """
  def update_schema(dataset_id, schema) do
    GenServer.call(__MODULE__, {:update_schema, dataset_id, schema})
  end

  # Server callbacks

  @impl true
  def init(:ok) do
    Logger.info("Starting schema inference engine")

    # Initialize with empty state
    {:producer_consumer, %{schemas: %{}},
     subscribe_to: [{IngestionService.Pipeline.Processor, max_demand: 10}]}
  end

  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Inferring schema for #{length(events)} events")

    # Process each event to infer schema
    processed_events =
      Enum.map(events, fn event ->
        try do
          new_event = infer_event_schema(event, state)
          {:ok, new_event}
        rescue
          e ->
            Logger.error("Error inferring schema: #{inspect(e)}")
            {:error, event, e}
        end
      end)

    # Separate successful and failed events
    {successful, failed} =
      Enum.split_with(processed_events, fn
        {:ok, _} -> true
        {:error, _, _} -> false
      end)

    # Extract successful events
    successful_events = Enum.map(successful, fn {:ok, event} -> event end)

    # Log failures
    Enum.each(failed, fn {:error, event, error} ->
      Logger.warn("Failed to infer schema for #{event.file_path}: #{inspect(error)}")
    end)

    # Calculate updated state with new schemas
    updated_state = update_state_schemas(successful_events, state)

    # Forward successful events
    {:noreply, successful_events, updated_state}
  end

  @impl true
  def handle_call({:get_schema, dataset_id}, _from, state) do
    schema = Map.get(state.schemas, dataset_id)
    {:reply, schema, [], state}
  end

  @impl true
  def handle_call({:update_schema, dataset_id, schema}, _from, state) do
    new_schemas = Map.put(state.schemas, dataset_id, schema)
    new_state = %{state | schemas: new_schemas}
    {:reply, :ok, [], new_state}
  end

  # Private functions

  # Infers schema from a data sample
  defp do_inference(data_sample) do
    # Analyze data types, patterns, and statistical properties
    fields =
      data_sample
      |> Enum.map(fn {field, values} ->
        {field, infer_field_properties(field, values)}
      end)

    %{
      fields: Map.new(fields),
      inferred_at: DateTime.utc_now(),
      confidence: calculate_confidence(fields)
    }
  end

  # Infers properties for a single field
  defp infer_field_properties(field, values) do
    # Detect data types, patterns, ranges, etc.
    %{
      type: detect_type(values),
      nullable: Enum.any?(values, &is_nil/1),
      unique_ratio: unique_ratio(values),
      stats: basic_statistics(values),
      patterns: detect_patterns(field, values)
    }
  end

  # Detects the likely type of a field based on sample values
  defp detect_type(values) do
    # Remove nil values for type detection
    clean_values = Enum.reject(values, &is_nil/1)

    cond do
      # Empty list, can't determine type
      Enum.empty?(clean_values) ->
        :unknown

      # Check if all values are numeric
      Enum.all?(clean_values, &is_number/1) ->
        if Enum.all?(clean_values, &(is_integer(&1) || floor(&1) == &1)) do
          :integer
        else
          :float
        end

      # Check if all values could be dates/timestamps
      Enum.all?(clean_values, &is_binary/1) &&
          Enum.all?(clean_values, &looks_like_timestamp?/1) ->
        :timestamp

      # Check if all values are binaries
      Enum.all?(clean_values, &is_binary/1) ->
        :string

      # Check if all values are booleans
      Enum.all?(clean_values, &is_boolean/1) ->
        :boolean

      # Mixed types, need further analysis
      true ->
        determine_dominant_type(clean_values)
    end
  end

  # Determines if a string looks like a timestamp
  defp looks_like_timestamp?(str) do
    # Simple regex patterns for common timestamp formats
    date_pattern = ~r/^\d{4}[-\/]\d{1,2}[-\/]\d{1,2}$/
    datetime_pattern = ~r/^\d{4}[-\/]\d{1,2}[-\/]\d{1,2}[ T]\d{1,2}:\d{1,2}(:\d{1,2})?$/
    unix_timestamp_pattern = ~r/^\d{10,13}$/

    String.match?(str, date_pattern) ||
      String.match?(str, datetime_pattern) ||
      String.match?(str, unix_timestamp_pattern)
  end

  # Determines the dominant type in a mixed-type list
  defp determine_dominant_type(values) do
    # Count occurrences of each type
    type_counts =
      Enum.reduce(values, %{}, fn value, acc ->
        type =
          cond do
            is_integer(value) ->
              :integer

            is_float(value) ->
              :float

            is_binary(value) ->
              if looks_like_timestamp?(value), do: :timestamp, else: :string

            is_boolean(value) ->
              :boolean

            true ->
              :other
          end

        Map.update(acc, type, 1, &(&1 + 1))
      end)

    # Find the most common type
    {dominant_type, _count} =
      type_counts
      |> Enum.sort_by(fn {_type, count} -> count end, :desc)
      |> List.first()

    dominant_type
  end

  # Calculates the unique ratio of values
  defp unique_ratio(values) do
    if Enum.empty?(values) do
      0.0
    else
      unique_count = values |> Enum.uniq() |> Enum.count()
      unique_count / Enum.count(values)
    end
  end

  # Calculates basic statistics for a field
  defp basic_statistics(values) do
    numeric_values = Enum.filter(values, &is_number/1)

    if Enum.empty?(numeric_values) do
      # For non-numeric values
      %{
        count: Enum.count(values),
        unique: values |> Enum.uniq() |> Enum.count(),
        null_count: Enum.count(values, &is_nil/1)
      }
    else
      # For numeric values
      sorted = Enum.sort(numeric_values)
      count = Enum.count(numeric_values)
      sum = Enum.sum(numeric_values)

      %{
        count: count,
        min: List.first(sorted),
        max: List.last(sorted),
        mean: sum / count,
        median: median(sorted),
        unique: numeric_values |> Enum.uniq() |> Enum.count(),
        null_count: Enum.count(values, &is_nil/1)
      }
    end
  end

  # Calculates the median of a sorted list
  defp median(sorted) do
    count = Enum.count(sorted)

    if count == 0 do
      nil
    else
      mid = div(count, 2)

      if rem(count, 2) == 0 do
        (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
      else
        Enum.at(sorted, mid)
      end
    end
  end

  # Detects patterns in field values (especially useful for strings)
  defp detect_patterns(field, values) do
    string_values = Enum.filter(values, &is_binary/1)

    if Enum.empty?(string_values) do
      %{type: :no_pattern}
    else
      # Look for field-specific patterns
      cond do
        looks_like_email_field?(field) ->
          %{
            type: :email,
            conformity: email_conformity_ratio(string_values)
          }

        looks_like_phone_field?(field) ->
          %{
            type: :phone,
            conformity: phone_conformity_ratio(string_values)
          }

        looks_like_date_field?(field) ->
          %{
            type: :date,
            formats: detect_date_formats(string_values),
            conformity: date_conformity_ratio(string_values)
          }

        true ->
          # Generic string analysis
          %{
            type: :generic,
            avg_length: average_string_length(string_values),
            has_numbers: contains_numbers?(string_values),
            has_special: contains_special_chars?(string_values)
          }
      end
    end
  end

  # Checks if a field name suggests it contains emails
  defp looks_like_email_field?(field) do
    field = field |> to_string() |> String.downcase()

    String.contains?(field, "email") ||
      String.contains?(field, "e-mail") ||
      String.contains?(field, "mail")
  end

  # Checks if a field name suggests it contains phone numbers
  defp looks_like_phone_field?(field) do
    field = field |> to_string() |> String.downcase()

    String.contains?(field, "phone") ||
      String.contains?(field, "mobile") ||
      String.contains?(field, "cell") ||
      String.contains?(field, "tel")
  end

  # Checks if a field name suggests it contains dates
  defp looks_like_date_field?(field) do
    field = field |> to_string() |> String.downcase()

    String.contains?(field, "date") ||
      String.contains?(field, "time") ||
      String.contains?(field, "day") ||
      String.contains?(field, "month") ||
      String.contains?(field, "year") ||
      String.contains?(field, "timestamp")
  end

  # Calculates the ratio of values that look like emails
  defp email_conformity_ratio(values) do
    email_pattern = ~r/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
    conforming = Enum.count(values, &String.match?(&1, email_pattern))
    conforming / Enum.count(values)
  end

  # Calculates the ratio of values that look like phone numbers
  defp phone_conformity_ratio(values) do
    # This is a simplified pattern - real phone validation is more complex
    phone_pattern = ~r/^[\d\s\+\-\(\)\.]{7,20}$/
    conforming = Enum.count(values, &String.match?(&1, phone_pattern))
    conforming / Enum.count(values)
  end

  # Detects common date formats in the values
  defp detect_date_formats(values) do
    # Common date format patterns
    formats = [
      {~r/^\d{4}-\d{2}-\d{2}$/, "yyyy-MM-dd"},
      {~r/^\d{2}\/\d{2}\/\d{4}$/, "MM/dd/yyyy"},
      {~r/^\d{4}\/\d{2}\/\d{2}$/, "yyyy/MM/dd"},
      {~r/^\d{2}-\d{2}-\d{4}$/, "MM-dd-yyyy"},
      {~r/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/, "ISO8601"},
      {~r/^\d{10,13}$/, "Unix timestamp"}
    ]

    # Count matches for each format
    format_counts =
      Enum.reduce(formats, %{}, fn {pattern, format}, acc ->
        count = Enum.count(values, &String.match?(&1, pattern))
        if count > 0, do: Map.put(acc, format, count), else: acc
      end)

    # Return detected formats sorted by frequency
    format_counts
    |> Enum.sort_by(fn {_format, count} -> count end, :desc)
    |> Enum.map(fn {format, _count} -> format end)
  end

  # Calculates the ratio of values that look like dates
  defp date_conformity_ratio(values) do
    # Combined pattern for various date formats
    date_pattern =
      ~r/^(\d{4}-\d{2}-\d{2}|\d{2}\/\d{2}\/\d{4}|\d{4}\/\d{2}\/\d{2}|\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\d{10,13})$/

    conforming = Enum.count(values, &String.match?(&1, date_pattern))
    conforming / Enum.count(values)
  end

  # Calculates the average string length
  defp average_string_length(values) do
    if Enum.empty?(values) do
      0
    else
      total_length = Enum.reduce(values, 0, fn str, acc -> String.length(str) + acc end)
      total_length / Enum.count(values)
    end
  end

  # Checks if any strings contain numbers
  defp contains_numbers?(values) do
    Enum.any?(values, &String.match?(&1, ~r/\d/))
  end

  # Checks if any strings contain special characters
  defp contains_special_chars?(values) do
    Enum.any?(values, &String.match?(&1, ~r/[^\w\s]/))
  end

  # Calculates confidence score for inferred schema
  defp calculate_confidence(fields) do
    # Average the confidence for each field
    field_confidences =
      Enum.map(fields, fn {_field, properties} ->
        calculate_field_confidence(properties)
      end)

    if Enum.empty?(field_confidences) do
      0.0
    else
      Enum.sum(field_confidences) / Enum.count(field_confidences)
    end
  end

  # Calculates confidence for a single field inference
  defp calculate_field_confidence({_field, properties}) do
    # Base confidence starts at 0.5
    base_confidence = 0.5

    # Adjust based on properties
    type_confidence =
      case properties.type do
        :unknown -> 0.0
        :string -> 0.7
        :integer -> 0.9
        :float -> 0.9
        :timestamp -> 0.8
        :boolean -> 0.95
        _ -> 0.5
      end

    # If we have patterns with high conformity, boost confidence
    pattern_boost =
      case properties[:patterns] do
        %{conformity: conf} when conf > 0.8 -> 0.2
        %{conformity: conf} when conf > 0.5 -> 0.1
        _ -> 0.0
      end

    # Adjust for uniqueness - very high or very low is more confident
    uniqueness_boost =
      if properties.unique_ratio > 0.9 or properties.unique_ratio < 0.1 do
        0.1
      else
        0.0
      end

    # Calculate final confidence
    final_confidence = base_confidence + type_confidence + pattern_boost + uniqueness_boost

    # Cap at 1.0
    min(final_confidence, 1.0)
  end

  # Merges two schemas, preferring the newer one but preserving information
  defp merge_schemas(existing, inferred) do
    # Merge fields
    merged_fields =
      Map.merge(existing.fields, inferred.fields, fn _k, v1, v2 ->
        # For each field, prefer the schema with higher confidence
        if v2.confidence > v1.confidence do
          # Keep some historical information
          v2
          |> Map.put(:previous_type, v1.type)
          |> Map.put(:changed_at, DateTime.utc_now())
        else
          v1
        end
      end)

    # Build merged schema
    %{
      fields: merged_fields,
      inferred_at: inferred.inferred_at,
      previous_inference: existing.inferred_at,
      confidence: calculate_confidence(merged_fields |> Map.to_list()),
      version: (existing[:version] || 0) + 1
    }
  end

  # Infers schema for a single event
  defp infer_event_schema(event, state) do
    # Extract dataset ID from the event
    dataset_id = extract_dataset_id(event)

    # Get existing schema if available
    existing_schema = Map.get(state.schemas, dataset_id)

    # Create column-based sample for inference
    data_sample = prepare_data_sample(event.data)

    # Infer schema and merge with existing
    schema = infer_schema(data_sample, existing_schema)

    # Add schema information to the event
    Map.put(event, :schema, schema)
  end

  # Updates state with new schemas from processed events
  defp update_state_schemas(events, state) do
    # Extract schemas from events and index by dataset_id
    new_schemas =
      Enum.reduce(events, %{}, fn event, acc ->
        dataset_id = extract_dataset_id(event)
        Map.put(acc, dataset_id, event.schema)
      end)

    # Merge with existing schemas
    updated_schemas = Map.merge(state.schemas, new_schemas)

    # Return updated state
    %{state | schemas: updated_schemas}
  end

  # Extracts a dataset ID from an event
  defp extract_dataset_id(event) do
    # Generate a dataset ID based on file path or other metadata
    # This is simplified - you may want a more sophisticated approach
    case event do
      %{metadata: %{dataset_id: id}} when not is_nil(id) ->
        id

      %{file_path: path} when not is_nil(path) ->
        # Use file path pattern as dataset ID
        path
        |> Path.basename()
        |> Path.rootname()
        |> normalize_dataset_name()

      _ ->
        # Fallback to a timestamp-based ID
        "unknown_dataset_#{System.system_time(:second)}"
    end
  end

  # Normalize a dataset name (e.g., from filename)
  defp normalize_dataset_name(name) do
    name
    |> String.downcase()
    # Remove digits
    |> String.replace(~r/\d+/, "")
    # Replace non-word chars with underscore
    |> String.replace(~r/[^\w]/, "_")
    # Replace multiple underscores with single
    |> String.replace(~r/_+/, "_")
    # Trim leading/trailing underscores
    |> String.trim("_")
  end

  # Prepares data in the right format for schema inference
  defp prepare_data_sample(data) do
    cond do
      is_list(data) && Enum.all?(data, &is_map/1) ->
        # List of maps - convert to column format
        rows_to_columns(data)

      is_list(data) && Enum.all?(data, &is_list/1) ->
        # List of lists (e.g. CSV data) - assume first row is headers
        [headers | rows] = data

        rows_with_headers =
          Enum.map(rows, fn row ->
            Enum.zip(headers, row) |> Map.new()
          end)

        rows_to_columns(rows_with_headers)

      is_map(data) ->
        # Already a map, assume it's in column format
        data

      true ->
        # Unsupported format
        Logger.warn("Unsupported data format for schema inference")
        %{}
    end
  end

  # Converts row-based data to column-based
  defp rows_to_columns(rows) do
    # Get all unique keys
    all_keys =
      rows
      |> Enum.flat_map(&Map.keys/1)
      |> Enum.uniq()

    # Build columns map
    Enum.reduce(all_keys, %{}, fn key, columns ->
      # Extract all values for this key
      values = Enum.map(rows, &Map.get(&1, key))
      Map.put(columns, key, values)
    end)
  end
end
