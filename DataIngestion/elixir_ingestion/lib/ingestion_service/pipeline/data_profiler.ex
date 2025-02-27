defmodule IngestionService.Pipeline.DataProfiler do
  use GenStage
  require Logger

  @moduledoc """
  A GenStage consumer-producer that performs comprehensive data profiling.
  Subscribes to the Validator and provides quality metrics to downstream stages.

  The profiler evaluates data across multiple quality dimensions:

  1. Completeness - Measures the presence of required data
  2. Consistency - Checks if data follows expected patterns
  3. Timeliness - Assesses how current data is relative to expectations
  4. Validity - Confirms data meets domain-specific rules
  5. Accuracy - Estimates how accurate data is using statistical methods
  6. Uniqueness - Measures duplication levels in the data
  """

  # Client API

  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @doc """
  Profiles a dataset and returns quality metrics
  """
  def profile_dataset(dataset, options \\ []) do
    # This allows profiling from outside the pipeline if needed
    # Useful for testing or ad-hoc profiling
    quality_metrics = %{
      completeness: calculate_completeness(dataset),
      consistency: check_consistency(dataset),
      timeliness: assess_timeliness(dataset, options),
      validity: check_validity(dataset),
      accuracy: estimate_accuracy(dataset),
      uniqueness: measure_uniqueness(dataset)
    }

    # Calculate overall quality score
    overall_score = calculate_overall_quality(quality_metrics)

    # Return all metrics with overall score
    Map.put(quality_metrics, :overall_score, overall_score)
  end

  # Server callbacks

  @impl true
  def init(:ok) do
    Logger.info("Starting data profiler")

    {:consumer_producer, %{},
     subscribe_to: [{IngestionService.Pipeline.Validator, max_demand: 10}]}
  end

  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Profiling #{length(events)} events")

    # Process events in parallel using Flow for better performance
    profiled_events =
      Flow.from_enumerable(events)
      |> Flow.map(&profile_event/1)
      |> Enum.to_list()

    {:noreply, profiled_events, state}
  end

  # Private functions

  defp profile_event(event) do
    # Add telemetry for profiling
    start_time = System.monotonic_time()

    try do
      # Profile the data
      quality_metrics = profile_dataset(event.data)

      # Track processing time
      duration = System.monotonic_time() - start_time

      :telemetry.execute(
        [:ingestion_service, :pipeline, :profile],
        %{duration: duration, success: true},
        %{event_id: event.id, file_path: event.file_path}
      )

      # Add quality profile to the event
      Map.put(event, :quality_profile, quality_metrics)
    rescue
      error ->
        # Log the error
        Logger.error("Error profiling data for #{event.file_path}: #{inspect(error)}")

        # Track the failure
        duration = System.monotonic_time() - start_time

        :telemetry.execute(
          [:ingestion_service, :pipeline, :profile],
          %{duration: duration, success: false},
          %{event_id: event.id, file_path: event.file_path, error: inspect(error)}
        )

        # Return event with error
        Map.put(event, :quality_profile, %{error: inspect(error), overall_score: 0.0})
    end
  end

  # Quality metric calculations

  defp calculate_completeness(data) when is_list(data) do
    # For a list of maps (rows)
    field_counts = count_field_presences(data)
    total_fields = length(data) * Map.size(field_counts)

    if total_fields == 0 do
      0.0
    else
      sum_present = Enum.sum(Map.values(field_counts))
      sum_present / total_fields
    end
  end

  defp calculate_completeness(data) when is_map(data) do
    # For a single map or map of sheet data (Excel case)
    if Map.has_key?(data, "__struct__") do
      # Handle struct case specially
      calculate_struct_completeness(data)
    else
      # Either a single record or a map of datasets (sheets)
      cond do
        Enum.all?(data, fn {_k, v} -> is_list(v) end) ->
          # Map of datasets (e.g., Excel sheets)
          sheet_scores =
            Enum.map(data, fn {_sheet, records} ->
              calculate_completeness(records)
            end)

          # Average completeness across sheets
          Enum.sum(sheet_scores) / length(sheet_scores)

        true ->
          # Single record as a map
          non_nil_count = Enum.count(data, fn {_k, v} -> not is_nil(v) end)
          non_nil_count / map_size(data)
      end
    end
  end

  defp calculate_completeness(_), do: 0.0

  defp calculate_struct_completeness(struct) do
    # Extract fields from struct, ignoring special fields
    fields =
      struct
      |> Map.from_struct()
      |> Map.drop([:__meta__])

    # Count non-nil fields
    non_nil_count = Enum.count(fields, fn {_k, v} -> not is_nil(v) end)
    non_nil_count / map_size(fields)
  end

  defp count_field_presences(rows) do
    # Get all possible field names
    all_fields =
      rows
      |> Enum.flat_map(&Map.keys/1)
      |> Enum.uniq()

    # Initialize counts
    initial_counts = Map.new(all_fields, fn field -> {field, 0} end)

    # Count presences
    Enum.reduce(rows, initial_counts, fn row, counts ->
      Enum.reduce(row, counts, fn {field, value}, acc ->
        if not is_nil(value) and value != "" do
          Map.update!(acc, field, &(&1 + 1))
        else
          acc
        end
      end)
    end)
  end

  defp check_consistency(data) when is_list(data) do
    # Skip empty lists
    if Enum.empty?(data) do
      0.0
    else
      # For each field, check consistency of values
      all_fields =
        data
        |> Enum.flat_map(&Map.keys/1)
        |> Enum.uniq()

      field_scores =
        Enum.map(all_fields, fn field ->
          values = Enum.map(data, &Map.get(&1, field))
          check_field_consistency(field, values)
        end)

      # Average field consistency scores
      Enum.sum(field_scores) / length(field_scores)
    end
  end

  defp check_consistency(data) when is_map(data) do
    # For a map of datasets (sheets)
    if Enum.all?(data, fn {_k, v} -> is_list(v) end) do
      # Calculate consistency for each sheet
      sheet_scores =
        Enum.map(data, fn {_sheet, records} ->
          check_consistency(records)
        end)

      # Average consistency across sheets
      Enum.sum(sheet_scores) / length(sheet_scores)
    else
      # Single record - return 1.0 (consistent with itself)
      1.0
    end
  end

  defp check_consistency(_), do: 0.0

  defp check_field_consistency(field, values) do
    # Remove nils for consistency check
    clean_values = Enum.reject(values, &is_nil/1)

    if Enum.empty?(clean_values) do
      # No values means no inconsistency
      1.0
    else
      # Different checks based on inferred field type
      cond do
        # Numeric fields - check range and distribution
        Enum.all?(clean_values, &is_number/1) ->
          numeric_consistency(clean_values)

        # String fields - check patterns
        Enum.all?(clean_values, &is_binary/1) ->
          string_consistency(field, clean_values)

        # Date fields - check formatting
        looks_like_date_field?(field) && Enum.all?(clean_values, &is_binary/1) ->
          date_consistency(clean_values)

        # Mixed or other types
        true ->
          # Fallback to type consistency
          type_consistency(clean_values)
      end
    end
  end

  defp numeric_consistency(values) do
    # Calculate coefficient of variation (CV) as a measure of dispersion
    {mean, stddev} = calculate_mean_stddev(values)

    if mean == 0 do
      # Can't calculate CV if mean is 0
      1.0
    else
      cv = stddev / mean

      # Convert CV to a consistency score (0 to 1)
      # Lower CV means higher consistency
      consistency = :math.exp(-cv)
      max(0.0, min(1.0, consistency))
    end
  end

  defp calculate_mean_stddev(values) do
    n = length(values)
    mean = Enum.sum(values) / n

    # Calculate variance
    variance =
      Enum.reduce(values, 0, fn x, acc ->
        diff = x - mean
        acc + diff * diff
      end) / n

    # Return mean and standard deviation
    {mean, :math.sqrt(variance)}
  end

  defp string_consistency(field, values) do
    # Use different approaches based on the field name
    cond do
      looks_like_email_field?(field) ->
        # Email pattern conformity
        email_pattern = ~r/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
        conformity_ratio(values, email_pattern)

      looks_like_phone_field?(field) ->
        # Phone pattern conformity
        phone_pattern = ~r/^[\d\s\+\-\(\)\.]{7,20}$/
        conformity_ratio(values, phone_pattern)

      true ->
        # General string consistency check
        # Look at length variation and character types

        # Calculate coefficient of variation for lengths
        lengths = Enum.map(values, &String.length/1)
        {mean_length, stddev_length} = calculate_mean_stddev(lengths)

        if mean_length == 0 do
          # Empty strings
          0.5
        else
          # Length consistency - higher coefficient means lower consistency
          cv_length = stddev_length / mean_length
          length_consistency = :math.exp(-cv_length)

          # Character type consistency
          char_type_consistency(values)

          # Combine both scores
          (length_consistency + char_type_consistency) / 2
        end
    end
  end

  defp conformity_ratio(values, pattern) do
    matching = Enum.count(values, &String.match?(&1, pattern))
    matching / length(values)
  end

  defp character_type_consistency(strings) do
    # Detect what type of strings we're dealing with
    patterns = [
      {~r/^[a-zA-Z]+$/, :alpha},
      {~r/^\d+$/, :numeric},
      {~r/^[a-zA-Z0-9]+$/, :alphanumeric},
      {~r/^[a-zA-Z\s]+$/, :alpha_with_spaces},
      {~r/^[a-zA-Z0-9\s]+$/, :alphanumeric_with_spaces}
    ]

    # Check each pattern and find the most common
    pattern_counts =
      Enum.map(patterns, fn {pattern, type} ->
        count = Enum.count(strings, &String.match?(&1, pattern))
        {type, count}
      end)

    # Get the most common pattern
    {_type, max_count} = Enum.max_by(pattern_counts, fn {_type, count} -> count end)

    # Return ratio of strings matching the dominant pattern
    max_count / length(strings)
  end

  defp date_consistency(values) do
    # Try to identify the date format
    formats = detect_date_formats(values)

    if Enum.empty?(formats) do
      # No consistent date format found
      0.0
    else
      # Calculate what percentage of dates match the most common format
      {most_common_format, _} = formats |> List.first()

      # Count dates matching this format
      matching =
        Enum.count(values, fn date_str ->
          match_date_format?(date_str, most_common_format)
        end)

      matching / length(values)
    end
  end

  defp match_date_format?(date_str, format) do
    case format do
      "yyyy-MM-dd" -> String.match?(date_str, ~r/^\d{4}-\d{2}-\d{2}$/)
      "MM/dd/yyyy" -> String.match?(date_str, ~r/^\d{2}\/\d{2}\/\d{4}$/)
      "yyyy/MM/dd" -> String.match?(date_str, ~r/^\d{4}\/\d{2}\/\d{2}$/)
      "MM-dd-yyyy" -> String.match?(date_str, ~r/^\d{2}-\d{2}-\d{4}$/)
      "ISO8601" -> String.match?(date_str, ~r/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)
      "Unix timestamp" -> String.match?(date_str, ~r/^\d{10,13}$/)
      _ -> false
    end
  end

  defp type_consistency(values) do
    # Count type occurrences
    type_counts =
      Enum.reduce(values, %{}, fn value, acc ->
        type = value_type(value)
        Map.update(acc, type, 1, &(&1 + 1))
      end)

    # Find most common type
    {_type, max_count} = Enum.max_by(type_counts, fn {_type, count} -> count end)

    # Return ratio of values of the dominant type
    max_count / length(values)
  end

  defp value_type(value) do
    cond do
      is_integer(value) -> :integer
      is_float(value) -> :float
      is_binary(value) -> :string
      is_boolean(value) -> :boolean
      is_nil(value) -> nil
      is_map(value) -> :map
      is_list(value) -> :list
      true -> :other
    end
  end

  defp assess_timeliness(data, options \\ []) when is_list(data) do
    # Find timestamp fields
    timestamp_fields = find_timestamp_fields(data)

    if Enum.empty?(timestamp_fields) do
      # No timestamp fields found
      nil
    else
      # For each timestamp field, calculate timeliness
      now = options[:reference_time] || DateTime.utc_now()

      field_scores =
        Enum.map(timestamp_fields, fn field ->
          timestamps = extract_timestamps(data, field)
          calculate_timestamp_timeliness(timestamps, now)
        end)

      # Return the maximum timeliness score (most recent field)
      Enum.max(field_scores, fn -> nil end)
    end
  end

  defp assess_timeliness(data, options) when is_map(data) do
    # For a map of datasets (sheets)
    if Enum.all?(data, fn {_k, v} -> is_list(v) end) do
      # Calculate timeliness for each sheet
      sheet_scores =
        Enum.map(data, fn {_sheet, records} ->
          assess_timeliness(records, options)
        end)

      # Filter out nil scores
      valid_scores = Enum.reject(sheet_scores, &is_nil/1)

      if Enum.empty?(valid_scores) do
        nil
      else
        # Return maximum timeliness across sheets
        Enum.max(valid_scores)
      end
    else
      # Single record - check if it has timestamp fields
      timestamp_fields =
        Enum.filter(data, fn {k, v} ->
          looks_like_date_field?(k) && (is_binary(v) || is_integer(v))
        end)

      if Enum.empty?(timestamp_fields) do
        nil
      else
        # Calculate timeliness of the most recent timestamp
        now = options[:reference_time] || DateTime.utc_now()

        timestamps =
          Enum.map(timestamp_fields, fn {_k, v} -> v end)
          |> Enum.filter(&(&1 != nil))

        calculate_timestamp_timeliness(timestamps, now)
      end
    end
  end

  defp assess_timeliness(_, _), do: nil

  defp find_timestamp_fields(data) do
    # Get a sample row
    sample = List.first(data)

    if is_nil(sample) do
      []
    else
      # Find fields that look like timestamps
      Enum.filter(Map.keys(sample), fn field ->
        looks_like_date_field?(field)
      end)
    end
  end

  defp extract_timestamps(data, field) do
    # Extract values for this field
    Enum.map(data, fn row ->
      timestamp = Map.get(row, field)

      if is_nil(timestamp) do
        nil
      else
        try_parse_timestamp(timestamp)
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp try_parse_timestamp(value) when is_binary(value) do
    # Try different timestamp formats
    cond do
      # ISO8601
      String.match?(value, ~r/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/) ->
        case DateTime.from_iso8601(value) do
          {:ok, datetime, _offset} -> datetime
          _ -> nil
        end

      # yyyy-MM-dd
      String.match?(value, ~r/^\d{4}-\d{2}-\d{2}$/) ->
        case Date.from_iso8601(value) do
          {:ok, date} ->
            # Convert to DateTime at midnight
            naive = %NaiveDateTime{
              year: date.year,
              month: date.month,
              day: date.day,
              hour: 0,
              minute: 0,
              second: 0
            }

            DateTime.from_naive!(naive, "Etc/UTC")

          _ ->
            nil
        end

      # Unix timestamp as string
      String.match?(value, ~r/^\d{10,13}$/) ->
        {unix_time, _} = Integer.parse(value)
        try_parse_timestamp(unix_time)

      # Other formats could be added here

      true ->
        nil
    end
  end

  defp try_parse_timestamp(value) when is_integer(value) do
    # Assume unix timestamp in seconds (10 digits) or milliseconds (13 digits)
    cond do
      value > 1_000_000_000_000 ->
        # Milliseconds
        DateTime.from_unix!(div(value, 1000))

      value > 1_000_000_000 ->
        # Seconds
        DateTime.from_unix!(value)

      true ->
        nil
    end
  rescue
    _ -> nil
  end

  defp try_parse_timestamp(_), do: nil

  defp calculate_timestamp_timeliness(timestamps, now) do
    if Enum.empty?(timestamps) do
      0.0
    else
      # Find the most recent timestamp
      most_recent = Enum.max_by(timestamps, &DateTime.to_unix/1, fn -> nil end)

      if is_nil(most_recent) do
        0.0
      else
        # Calculate age in days
        diff_seconds = DateTime.diff(now, most_recent)
        age_days = diff_seconds / 86400

        # Timeliness score decreases with age
        # 1.0 for today, 0.5 for ~30 days old, approaching 0 as older
        :math.exp(-age_days / 30)
      end
    end
  end

  defp check_validity(data) when is_list(data) do
    # Skip empty lists
    if Enum.empty?(data) do
      1.0
    else
      # For each field, check validity
      all_fields =
        data
        |> Enum.flat_map(&Map.keys/1)
        |> Enum.uniq()

      field_scores =
        Enum.map(all_fields, fn field ->
          values = Enum.map(data, &Map.get(&1, field))
          check_field_validity(field, values)
        end)

      # Average field validity scores
      Enum.sum(field_scores) / length(field_scores)
    end
  end

  defp check_validity(data) when is_map(data) do
    # For a map of datasets (sheets)
    if Enum.all?(data, fn {_k, v} -> is_list(v) end) do
      # Calculate validity for each sheet
      sheet_scores =
        Enum.map(data, fn {_sheet, records} ->
          check_validity(records)
        end)

      # Average validity across sheets
      Enum.sum(sheet_scores) / length(sheet_scores)
    else
      # Single record - all fields valid by default
      1.0
    end
  end

  defp check_validity(_), do: 1.0

  defp check_field_validity(field, values) do
    # Remove nils for validity check
    clean_values = Enum.reject(values, &is_nil/1)

    if Enum.empty?(clean_values) do
      # No values means no invalidity
      1.0
    else
      # Different validation based on field name and inferred type
      cond do
        # Email validation
        looks_like_email_field?(field) && Enum.all?(clean_values, &is_binary/1) ->
          email_pattern = ~r/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
          conformity_ratio(clean_values, email_pattern)

        # Phone validation
        looks_like_phone_field?(field) && Enum.all?(clean_values, &is_binary/1) ->
          phone_pattern = ~r/^[\d\s\+\-\(\)\.]{7,20}$/
          conformity_ratio(clean_values, phone_pattern)

        # Date validation
        looks_like_date_field?(field) && Enum.all?(clean_values, &is_binary/1) ->
          # Check if dates are parseable
          parseable =
            Enum.count(clean_values, fn v ->
              try_parse_timestamp(v) != nil
            end)

          parseable / length(clean_values)

        # Numeric validation - check for outliers
        Enum.all?(clean_values, &is_number/1) ->
          check_numeric_validity(clean_values)

        # String validation - check for suspect characters
        Enum.all?(clean_values, &is_binary/1) ->
          check_string_validity(clean_values)

        # Default - all valid
        true ->
          1.0
      end
    end
  end

  defp check_numeric_validity(values) do
    # Calculate statistics
    {mean, stddev} = calculate_mean_stddev(values)

    # Check for outliers (more than 3 standard deviations from mean)
    outliers =
      Enum.count(values, fn x ->
        abs(x - mean) > 3 * stddev
      end)

    # Return ratio of non-outliers
    1.0 - outliers / length(values)
  end

  defp check_string_validity(values) do
    # Check for strings with suspect patterns
    suspect_patterns = [
      # Leading/trailing whitespace
      ~r/^\s+|\s+$/,
      # Null byte
      ~r/\x00/,
      # Non-printable ASCII
      ~r/[^\x20-\x7E]/
    ]

    # Count strings matching any suspect pattern
    suspect_count =
      Enum.count(values, fn str ->
        Enum.any?(suspect_patterns, &String.match?(str, &1))
      end)

    # Return ratio of valid strings
    1.0 - suspect_count / length(values)
  end

  defp estimate_accuracy(data) when is_list(data) do
    # This is a very rough estimate as true accuracy requires reference data
    # We'll use a combination of completeness, consistency and validity

    completeness = calculate_completeness(data)
    consistency = check_consistency(data)
    validity = check_validity(data)

    # Weighted average of these metrics
    0.4 * completeness + 0.3 * consistency + 0.3 * validity
  end

  defp estimate_accuracy(data) when is_map(data) do
    # For a map of datasets (sheets)
    if Enum.all?(data, fn {_k, v} -> is_list(v) end) do
      # Calculate accuracy for each sheet
      sheet_scores =
        Enum.map(data, fn {_sheet, records} ->
          estimate_accuracy(records)
        end)

      # Average accuracy across sheets
      Enum.sum(sheet_scores) / length(sheet_scores)
    else
      # Single record - use a simpler metric
      # Count fields with non-nil, non-empty values
      valid_fields =
        Enum.count(data, fn {_k, v} ->
          not is_nil(v) and (not is_binary(v) or v != "")
        end)

      valid_fields / map_size(data)
    end
  end

  # Default middle value when we can't estimate
  defp estimate_accuracy(_), do: 0.5

  defp measure_uniqueness(data) when is_list(data) do
    # Skip empty lists
    if Enum.empty?(data) do
      1.0
    else
      # For each field, measure uniqueness
      all_fields =
        data
        |> Enum.flat_map(&Map.keys/1)
        |> Enum.uniq()

      field_scores =
        Enum.map(all_fields, fn field ->
          values = Enum.map(data, &Map.get(&1, field))
          field_uniqueness(values)
        end)

      # Average field uniqueness scores
      Enum.sum(field_scores) / length(field_scores)
    end
  end

  defp measure_uniqueness(data) when is_map(data) do
    # For a map of datasets (sheets)
    if Enum.all?(data, fn {_k, v} -> is_list(v) end) do
      # Calculate uniqueness for each sheet
      sheet_scores =
        Enum.map(data, fn {_sheet, records} ->
          measure_uniqueness(records)
        end)

      # Average uniqueness across sheets
      Enum.sum(sheet_scores) / length(sheet_scores)
    else
      # Single record - all values are unique (to themselves)
      1.0
    end
  end

  defp measure_uniqueness(_), do: 1.0

  defp field_uniqueness(values) do
    # Remove nils for uniqueness check
    clean_values = Enum.reject(values, &is_nil/1)

    if Enum.empty?(clean_values) do
      # Empty set is uniquely empty
      1.0
    else
      # Count unique values
      unique_count = clean_values |> Enum.uniq() |> length()
      unique_count / length(clean_values)
    end
  end

  # Helper functions for field typing

  defp looks_like_email_field?(field) do
    field = to_string(field) |> String.downcase()

    String.contains?(field, "email") ||
      String.contains?(field, "e-mail") ||
      String.contains?(field, "mail")
  end

  defp looks_like_phone_field?(field) do
    field = to_string(field) |> String.downcase()

    String.contains?(field, "phone") ||
      String.contains?(field, "mobile") ||
      String.contains?(field, "cell") ||
      String.contains?(field, "tel")
  end

  defp looks_like_date_field?(field) do
    field = to_string(field) |> String.downcase()

    String.contains?(field, "date") ||
      String.contains?(field, "time") ||
      String.contains?(field, "day") ||
      String.contains?(field, "month") ||
      String.contains?(field, "year") ||
      String.contains?(field, "timestamp") ||
      String.contains?(field, "created") ||
      String.contains?(field, "updated") ||
      String.contains?(field, "modified")
  end

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
      Enum.map(formats, fn {pattern, format} ->
        count = Enum.count(values, &String.match?(&1, pattern))
        {format, count}
      end)

    # Return detected formats sorted by frequency, excluding zero counts
    format_counts
    |> Enum.filter(fn {_format, count} -> count > 0 end)
    |> Enum.sort_by(fn {_format, count} -> count end, :desc)
  end

  defp calculate_overall_quality(metrics) do
    # Define weights for each metric
    weights = %{
      completeness: 0.25,
      consistency: 0.2,
      timeliness: 0.15,
      validity: 0.15,
      accuracy: 0.15,
      uniqueness: 0.1
    }

    # Calculate weighted sum
    total_weight = 0
    weighted_sum = 0

    Enum.reduce(weights, {0, 0}, fn {metric, weight}, {sum, total_w} ->
      value = Map.get(metrics, metric)

      if is_nil(value) or not is_number(value) do
        # Skip metrics that are nil or not numeric
        {sum, total_w}
      else
        {sum + value * weight, total_w + weight}
      end
    end)
    |> case do
      # No valid metrics
      {_, 0} -> 0.0
      # Normalize by total weight
      {sum, total} -> sum / total
    end
  end
end
