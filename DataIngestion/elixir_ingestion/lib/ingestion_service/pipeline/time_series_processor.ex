defmodule IngestionService.Pipeline.TimeSeriesProcessor do
  use GenStage
  require Logger

  @moduledoc """
  A GenStage consumer-producer that provides specialized processing for time series data.
  Subscribes to the Data Profiler and handles time-based operations including:

  1. Timestamp normalization - Converts timestamps to a standard format
  2. Missing value imputation - Fills gaps in time series data
  3. Anomaly detection - Identifies outliers in time series data
  4. Time-based feature generation - Creates derived features based on time patterns
  """

  # Client API

  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @doc """
  Process a time series dataset with specialized handling.

  ## Options
  * `:time_column` - Name of the timestamp column (will be auto-detected if not provided)
  * `:value_columns` - List of columns containing values to process (all numeric columns by default)
  * `:target_frequency` - Target frequency for resampling ('hourly', 'daily', etc.)
  * `:imputation_strategy` - Strategy for filling missing values ('previous', 'mean', 'linear', etc.)
  * `:anomaly_detection` - Whether to detect anomalies (default: true)
  * `:feature_generation` - Whether to generate time-based features (default: true)
  """
  def process_time_series(data, options \\ []) do
    # This allows processing from outside the pipeline if needed
    # Useful for testing or ad-hoc processing

    # Detect time column if not specified
    time_column = options[:time_column] || detect_time_column(data)

    if is_nil(time_column) do
      Logger.warn("No time column found or specified for time series processing")
      %{data: data, time_series_info: %{error: "No time column found"}}
    else
      # Process the time series data
      try do
        # Sort data by timestamp
        sorted_data = sort_by_timestamp(data, time_column)

        # Detect value columns if not specified
        value_columns = options[:value_columns] || detect_value_columns(sorted_data, time_column)

        # Normalize timestamps
        normalized_data = normalize_timestamps(sorted_data, time_column, options)

        # Impute missing values if requested
        imputation_strategy = options[:imputation_strategy] || "previous"

        imputed_data =
          impute_missing_values(normalized_data, time_column, value_columns, imputation_strategy)

        # Detect anomalies if requested
        anomalies =
          if options[:anomaly_detection] != false do
            detect_anomalies(imputed_data, time_column, value_columns)
          else
            %{}
          end

        # Generate time-based features if requested
        processed_data =
          if options[:feature_generation] != false do
            generate_time_features(imputed_data, time_column)
          else
            imputed_data
          end

        # Calculate time series statistics
        time_stats = calculate_time_series_stats(processed_data, time_column, value_columns)

        # Return processed data with metadata
        %{
          data: processed_data,
          time_series_info: %{
            time_column: time_column,
            value_columns: value_columns,
            anomalies: anomalies,
            statistics: time_stats
          }
        }
      rescue
        error ->
          Logger.error("Error processing time series: #{inspect(error)}")

          %{
            data: data,
            time_series_info: %{
              error: "Processing error: #{inspect(error)}"
            }
          }
      end
    end
  end

  # Server callbacks

  @impl true
  def init(:ok) do
    Logger.info("Starting time series processor")

    {:consumer_producer, %{},
     subscribe_to: [{IngestionService.Pipeline.DataProfiler, max_demand: 10}]}
  end

  @impl true
  def handle_events(events, _from, state) do
    Logger.debug("Processing #{length(events)} time series events")

    # Process events in parallel using Flow for better performance
    processed_events =
      Flow.from_enumerable(events)
      |> Flow.map(&process_event/1)
      |> Enum.to_list()

    {:noreply, processed_events, state}
  end

  # Private functions

  defp process_event(event) do
    # Add telemetry for time series processing
    start_time = System.monotonic_time()

    try do
      # Check if the data looks like time series
      if has_time_series_characteristics?(event.data) do
        # Process as time series
        result = process_time_series(event.data)

        # Track processing time
        duration = System.monotonic_time() - start_time

        :telemetry.execute(
          [:ingestion_service, :pipeline, :time_series],
          %{duration: duration, success: true},
          %{event_id: event.id, file_path: event.file_path}
        )

        # Add time series info to the event
        Map.put(event, :time_series_info, result.time_series_info)
        |> Map.put(:data, result.data)
      else
        # Not time series data, just pass through
        event
      end
    rescue
      error ->
        # Log the error
        Logger.error("Error in time series processing for #{event.file_path}: #{inspect(error)}")

        # Track the failure
        duration = System.monotonic_time() - start_time

        :telemetry.execute(
          [:ingestion_service, :pipeline, :time_series],
          %{duration: duration, success: false},
          %{event_id: event.id, file_path: event.file_path, error: inspect(error)}
        )

        # Return event unchanged
        event
    end
  end

  # Time series detection

  defp has_time_series_characteristics?(data) when is_list(data) do
    # Skip empty lists
    if Enum.empty?(data) do
      false
    else
      # Check if there's at least one timestamp column
      time_column = detect_time_column(data)

      if is_nil(time_column) do
        false
      else
        # Check if there are numeric value columns
        value_columns = detect_value_columns(data, time_column)
        not Enum.empty?(value_columns)
      end
    end
  end

  defp has_time_series_characteristics?(data) when is_map(data) do
    # For a map of datasets (sheets)
    if Enum.all?(data, fn {_k, v} -> is_list(v) end) do
      # Check if any sheet has time series characteristics
      Enum.any?(data, fn {_sheet, records} ->
        has_time_series_characteristics?(records)
      end)
    else
      # Single record - not a time series
      false
    end
  end

  defp has_time_series_characteristics?(_), do: false

  defp detect_time_column(data) when is_list(data) do
    # Skip empty lists
    if Enum.empty?(data) do
      nil
    else
      # Get a sample row
      sample = List.first(data)

      # Find fields that look like timestamps
      timestamp_fields =
        Enum.filter(Map.keys(sample), fn field ->
          looks_like_date_field?(field) || has_timestamp_values?(data, field)
        end)

      # Return the first timestamp field, or nil if none found
      List.first(timestamp_fields)
    end
  end

  defp detect_time_column(_), do: nil

  defp has_timestamp_values?(data, field) do
    # Check if a significant number of values look like timestamps
    values =
      Enum.map(data, &Map.get(&1, field))
      |> Enum.reject(&is_nil/1)

    # Skip if no values
    if Enum.empty?(values) do
      false
    else
      # Check how many values can be parsed as timestamps
      parseable =
        Enum.count(values, fn value ->
          cond do
            is_binary(value) ->
              # Check common timestamp formats
              String.match?(value, ~r/^\d{4}-\d{2}-\d{2}(T|\s)\d{2}:\d{2}/) ||
                String.match?(value, ~r/^\d{4}-\d{2}-\d{2}$/) ||
                String.match?(value, ~r/^\d{2}\/\d{2}\/\d{4}$/) ||
                String.match?(value, ~r/^\d{10,13}$/)

            is_integer(value) && value > 1_000_000_000 ->
              # Likely a unix timestamp
              true

            true ->
              false
          end
        end)

      # Consider it a timestamp field if at least 80% of values look like timestamps
      parseable / length(values) >= 0.8
    end
  end

  defp detect_value_columns(data, time_column) when is_list(data) do
    # Skip empty lists
    if Enum.empty?(data) do
      []
    else
      # Get a sample row
      sample = List.first(data)

      # Find fields that have numeric values and aren't the time column
      numeric_fields =
        Enum.filter(Map.keys(sample), fn field ->
          field != time_column && has_numeric_values?(data, field)
        end)

      numeric_fields
    end
  end

  defp detect_value_columns(_, _), do: []

  defp has_numeric_values?(data, field) do
    # Check if a significant number of values are numeric
    values =
      Enum.map(data, &Map.get(&1, field))
      |> Enum.reject(&is_nil/1)

    # Skip if no values
    if Enum.empty?(values) do
      false
    else
      # Count numeric values
      numeric_count =
        Enum.count(values, fn value ->
          is_number(value) ||
            (is_binary(value) && String.match?(value, ~r/^-?\d+(\.\d+)?$/))
        end)

      # Consider it numeric if at least 80% of values are numeric
      numeric_count / length(values) >= 0.8
    end
  end

  # Time series processing

  defp sort_by_timestamp(data, time_column) when is_list(data) do
    # Parse timestamps and sort
    data
    |> Enum.map(fn row ->
      # Extract timestamp
      timestamp = Map.get(row, time_column)
      parsed_timestamp = parse_timestamp(timestamp)

      # Add parsed timestamp to the row for sorting
      Map.put(row, :__parsed_timestamp__, parsed_timestamp)
    end)
    |> Enum.sort_by(fn row ->
      # Sort by parsed timestamp
      row.__parsed_timestamp__
    end)
    |> Enum.map(fn row ->
      # Remove the parsed timestamp field
      Map.delete(row, :__parsed_timestamp__)
    end)
  end

  defp sort_by_timestamp(data, _), do: data

  defp parse_timestamp(value) do
    cond do
      is_nil(value) ->
        # Default to epoch start for nil values
        ~U[1970-01-01 00:00:00Z]

      is_binary(value) ->
        # Try different timestamp formats
        cond do
          # ISO8601
          String.match?(value, ~r/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/) ->
            case DateTime.from_iso8601(value) do
              {:ok, datetime, _} -> datetime
              _ -> default_timestamp(value)
            end

          # yyyy-MM-dd HH:mm:ss
          String.match?(value, ~r/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}/) ->
            case NaiveDateTime.from_iso8601(value <> ":00") do
              {:ok, naive} -> DateTime.from_naive!(naive, "Etc/UTC")
              _ -> default_timestamp(value)
            end

          # yyyy-MM-dd
          String.match?(value, ~r/^\d{4}-\d{2}-\d{2}$/) ->
            case Date.from_iso8601(value) do
              {:ok, date} ->
                %DateTime{
                  year: date.year,
                  month: date.month,
                  day: date.day,
                  hour: 0,
                  minute: 0,
                  second: 0,
                  microsecond: {0, 0},
                  std_offset: 0,
                  utc_offset: 0,
                  zone_abbr: "UTC",
                  time_zone: "Etc/UTC"
                }

              _ ->
                default_timestamp(value)
            end

          # MM/dd/yyyy
          String.match?(value, ~r/^\d{2}\/\d{2}\/\d{4}$/) ->
            [month, day, year] = String.split(value, "/")

            case Date.new(
                   String.to_integer(year),
                   String.to_integer(month),
                   String.to_integer(day)
                 ) do
              {:ok, date} ->
                %DateTime{
                  year: date.year,
                  month: date.month,
                  day: date.day,
                  hour: 0,
                  minute: 0,
                  second: 0,
                  microsecond: {0, 0},
                  std_offset: 0,
                  utc_offset: 0,
                  zone_abbr: "UTC",
                  time_zone: "Etc/UTC"
                }

              _ ->
                default_timestamp(value)
            end

          # Unix timestamp as string
          String.match?(value, ~r/^\d{10,13}$/) ->
            {unix_time, _} = Integer.parse(value)
            parse_timestamp(unix_time)

          true ->
            default_timestamp(value)
        end

      is_integer(value) ->
        # Assume unix timestamp in seconds (10 digits) or milliseconds (13 digits)
        cond do
          value > 1_000_000_000_000 ->
            # Milliseconds
            DateTime.from_unix!(div(value, 1000))

          value > 1_000_000_000 ->
            # Seconds
            DateTime.from_unix!(value)

          true ->
            default_timestamp(value)
        end

      true ->
        default_timestamp(value)
    end
  rescue
    _ -> default_timestamp(value)
  end

  defp default_timestamp(value) do
    # Return a default timestamp based on the string representation
    # This ensures consistent sorting even for unparseable timestamps
    case inspect(value) do
      str when is_binary(str) ->
        # Hash the string to get a consistent integer
        :erlang.phash2(str)
        |> DateTime.from_unix!()

      _ ->
        # Fallback to epoch start
        ~U[1970-01-01 00:00:00Z]
    end
  end

  defp normalize_timestamps(data, time_column, options) when is_list(data) do
    # Convert all timestamps to a standard format
    Enum.map(data, fn row ->
      timestamp = Map.get(row, time_column)
      parsed = parse_timestamp(timestamp)

      # Format the timestamp as ISO8601
      formatted = DateTime.to_iso8601(parsed)

      # Update the row with the normalized timestamp
      Map.put(row, time_column, formatted)
    end)
  end

  defp normalize_timestamps(data, _, _), do: data

  defp impute_missing_values(data, time_column, value_columns, strategy) when is_list(data) do
    # Skip if empty
    if Enum.empty?(data) do
      data
    else
      # Check for gaps in the time series
      sorted_data = sort_by_timestamp(data, time_column)

      # Get min and max timestamps
      timestamps = Enum.map(sorted_data, &Map.get(&1, time_column))
      parsed_timestamps = Enum.map(timestamps, &parse_timestamp/1)

      # Find minimum time gap to detect the time series frequency
      gaps = detect_time_gaps(parsed_timestamps)

      if Enum.empty?(gaps) || gaps.median_gap_seconds == 0 do
        # No significant gaps found, return data as is
        sorted_data
      else
        # Fill missing values based on detected frequency and strategy
        fill_time_gaps(sorted_data, time_column, value_columns, gaps, strategy)
      end
    end
  end

  defp impute_missing_values(data, _, _, _), do: data

  defp detect_time_gaps(timestamps) do
    # Calculate time differences between consecutive timestamps
    gaps =
      Enum.chunk_every(timestamps, 2, 1, :discard)
      |> Enum.map(fn [t1, t2] -> DateTime.diff(t2, t1) end)

    if Enum.empty?(gaps) do
      %{}
    else
      # Calculate statistics on gaps
      sorted_gaps = Enum.sort(gaps)

      %{
        min_gap_seconds: List.first(sorted_gaps),
        max_gap_seconds: List.last(sorted_gaps),
        median_gap_seconds: median(sorted_gaps),
        gap_frequency: most_common_gap(sorted_gaps),
        gaps: gaps
      }
    end
  end

  defp median(sorted_list) do
    count = length(sorted_list)
    middle = div(count, 2)

    if rem(count, 2) == 0 do
      (Enum.at(sorted_list, middle - 1) + Enum.at(sorted_list, middle)) / 2
    else
      Enum.at(sorted_list, middle)
    end
  end

  defp most_common_gap(gaps) do
    # Group gaps by value and find the most common
    gap_counts =
      Enum.reduce(gaps, %{}, fn gap, acc ->
        Map.update(acc, gap, 1, &(&1 + 1))
      end)

    {most_common, _count} =
      gap_counts
      |> Enum.max_by(fn {_gap, count} -> count end, fn -> {0, 0} end)

    most_common
  end

  defp fill_time_gaps(data, time_column, value_columns, gaps, strategy) do
    # Get frequency in seconds
    frequency = gaps.gap_frequency

    # Guard against very small frequencies (less than 1 second)
    if frequency < 1 do
      data
    else
      # Generate expected timestamps
      [first | _] = data
      first_timestamp = Map.get(first, time_column) |> parse_timestamp()

      last = List.last(data)
      last_timestamp = Map.get(last, time_column) |> parse_timestamp()

      # Generate a complete series of timestamps
      expected_timestamps =
        Stream.unfold(first_timestamp, fn current ->
          if DateTime.compare(current, last_timestamp) != :gt do
            next_time = DateTime.add(current, frequency)
            {current, next_time}
          else
            nil
          end
        end)
        |> Enum.to_list()

      # Convert data to a map keyed by timestamp for easier lookup
      timestamp_map =
        Enum.reduce(data, %{}, fn row, acc ->
          ts = Map.get(row, time_column) |> parse_timestamp()
          Map.put(acc, DateTime.to_iso8601(ts), row)
        end)

      # Fill missing values based on strategy
      Enum.map(expected_timestamps, fn ts ->
        iso_ts = DateTime.to_iso8601(ts)

        if Map.has_key?(timestamp_map, iso_ts) do
          # Use existing data
          Map.get(timestamp_map, iso_ts)
        else
          # Create a new row with imputed values
          imputed_row = %{time_column => iso_ts}

          # Fill value columns based on strategy
          Enum.reduce(value_columns, imputed_row, fn col, row ->
            imputed_value = impute_value(data, ts, col, strategy)
            Map.put(row, col, imputed_value)
          end)
        end
      end)
    end
  end

  defp impute_value(data, timestamp, column, strategy) do
    case strategy do
      "previous" ->
        # Use the previous value (forward fill)
        previous_rows =
          Enum.filter(data, fn row ->
            row_ts = Map.get(row, :time_column) |> parse_timestamp()
            DateTime.compare(row_ts, timestamp) == :lt
          end)

        if Enum.empty?(previous_rows) do
          nil
        else
          previous = List.last(previous_rows)
          Map.get(previous, column)
        end

      "mean" ->
        # Use the mean of all values
        values =
          Enum.map(data, &Map.get(&1, column))
          |> Enum.filter(&is_number/1)

        if Enum.empty?(values) do
          nil
        else
          Enum.sum(values) / length(values)
        end

      "linear" ->
        # Linear interpolation between nearest points
        # Find previous and next values
        previous_rows =
          Enum.filter(data, fn row ->
            row_ts = Map.get(row, :time_column) |> parse_timestamp()
            DateTime.compare(row_ts, timestamp) == :lt
          end)

        next_rows =
          Enum.filter(data, fn row ->
            row_ts = Map.get(row, :time_column) |> parse_timestamp()
            DateTime.compare(row_ts, timestamp) == :gt
          end)

        if Enum.empty?(previous_rows) || Enum.empty?(next_rows) do
          # Can't interpolate, use previous or next value
          cond do
            not Enum.empty?(previous_rows) ->
              previous = List.last(previous_rows)
              Map.get(previous, column)

            not Enum.empty?(next_rows) ->
              next = List.first(next_rows)
              Map.get(next, column)

            true ->
              nil
          end
        else
          # Interpolate between previous and next
          previous = List.last(previous_rows)
          prev_ts = Map.get(previous, :time_column) |> parse_timestamp()
          prev_val = Map.get(previous, column)

          next = List.first(next_rows)
          next_ts = Map.get(next, :time_column) |> parse_timestamp()
          next_val = Map.get(next, column)

          # Only interpolate if both values are numeric
          if is_number(prev_val) && is_number(next_val) do
            # Calculate time ratios
            total_diff = DateTime.diff(next_ts, prev_ts)
            current_diff = DateTime.diff(timestamp, prev_ts)
            ratio = current_diff / total_diff

            # Linear interpolation
            prev_val + (next_val - prev_val) * ratio
          else
            # Can't interpolate non-numeric values
            prev_val
          end
        end

      "zero" ->
        # Replace with zero
        0

      "null" ->
        # Leave as nil
        nil

      _ ->
        # Default to previous value
        previous_rows =
          Enum.filter(data, fn row ->
            row_ts = Map.get(row, :time_column) |> parse_timestamp()
            DateTime.compare(row_ts, timestamp) == :lt
          end)

        if Enum.empty?(previous_rows) do
          nil
        else
          previous = List.last(previous_rows)
          Map.get(previous, column)
        end
    end
  end

  defp detect_anomalies(data, time_column, value_columns) when is_list(data) do
    # Detect anomalies for each value column
    Enum.reduce(value_columns, %{}, fn column, acc ->
      # Extract values
      values =
        Enum.map(data, fn row ->
          value = Map.get(row, column)

          if is_binary(value) do
            # Try to convert string to number
            case Float.parse(value) do
              {num, _} -> num
              :error -> nil
            end
          else
            value
          end
        end)
        |> Enum.filter(&is_number/1)

      # Calculate statistics
      if Enum.empty?(values) do
        acc
      else
        stats = calculate_statistics(values)

        # Detect outliers using Z-score or IQR method
        anomalies =
          if stats.stddev > 0 do
            # Use Z-score method
            detect_anomalies_zscore(data, column, values, stats)
          else
            # Use IQR method
            detect_anomalies_iqr(data, column, values)
          end

        Map.put(acc, column, anomalies)
      end
    end)
  end

  defp detect_anomalies(data, _, _), do: %{}

  defp detect_anomalies_zscore(data, column, values, stats) do
    # Use Z-score to detect anomalies (|z| > 3 standard deviations)
    threshold = 3.0

    Enum.filter(data, fn row ->
      value = Map.get(row, column)

      if is_number(value) do
        z_score = (value - stats.mean) / stats.stddev
        abs(z_score) > threshold
      else
        false
      end
    end)
    |> Enum.map(fn row ->
      %{
        timestamp: Map.get(row, :time_column),
        value: Map.get(row, column),
        z_score: (Map.get(row, column) - stats.mean) / stats.stddev
      }
    end)
  end

  defp detect_anomalies_iqr(data, column, values) do
    # Use IQR method to detect anomalies (outside 1.5 * IQR)
    sorted = Enum.sort(values)
    q1 = percentile(sorted, 0.25)
    q3 = percentile(sorted, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    Enum.filter(data, fn row ->
      value = Map.get(row, column)

      if is_number(value) do
        value < lower_bound || value > upper_bound
      else
        false
      end
    end)
    |> Enum.map(fn row ->
      %{
        timestamp: Map.get(row, :time_column),
        value: Map.get(row, column),
        bounds: [lower_bound, upper_bound]
      }
    end)
  end

  defp percentile(sorted_list, p) do
    len = length(sorted_list)
    k = trunc(len * p)

    if k == 0 do
      List.first(sorted_list)
    else
      Enum.at(sorted_list, k - 1)
    end
  end

  defp calculate_statistics(values) do
    n = length(values)
    mean = Enum.sum(values) / n

    # Calculate variance and standard deviation
    variance =
      Enum.reduce(values, 0, fn x, acc ->
        diff = x - mean
        acc + diff * diff
      end) / n

    stddev = :math.sqrt(variance)

    # Calculate min, max, median
    sorted = Enum.sort(values)
    min = List.first(sorted)
    max = List.last(sorted)
    median_val = median(sorted)

    %{
      mean: mean,
      median: median_val,
      min: min,
      max: max,
      variance: variance,
      stddev: stddev
    }
  end

  defp generate_time_features(data, time_column) when is_list(data) do
    # Add time-based derived features
    Enum.map(data, fn row ->
      timestamp = Map.get(row, time_column)
      parsed = parse_timestamp(timestamp)

      # Add various time features
      row
      |> Map.put("#{time_column}_hour", parsed.hour)
      |> Map.put("#{time_column}_day", parsed.day)
      |> Map.put("#{time_column}_month", parsed.month)
      |> Map.put("#{time_column}_year", parsed.year)
      |> Map.put("#{time_column}_day_of_week", day_of_week(parsed))
      |> Map.put("#{time_column}_is_weekend", is_weekend?(parsed))
    end)
  end

  defp generate_time_features(data, _), do: data

  defp day_of_week(datetime) do
    # Calculate day of week (1 = Monday, 7 = Sunday)
    Date.day_of_week(datetime)
  end

  defp is_weekend?(datetime) do
    # Check if the date is a weekend (Saturday or Sunday)
    dow = day_of_week(datetime)
    dow == 6 || dow == 7
  end

  defp calculate_time_series_stats(data, time_column, value_columns) when is_list(data) do
    # Skip if empty
    if Enum.empty?(data) do
      %{}
    else
      # Get timestamps
      timestamps = Enum.map(data, &Map.get(&1, time_column))
      parsed_timestamps = Enum.map(timestamps, &parse_timestamp/1)

      # Calculate time range
      sorted_timestamps = Enum.sort(parsed_timestamps, DateTime)
      first_timestamp = List.first(sorted_timestamps)
      last_timestamp = List.last(sorted_timestamps)

      # Calculate duration in seconds
      duration_seconds = DateTime.diff(last_timestamp, first_timestamp)

      # Calculate time gaps
      gaps = detect_time_gaps(sorted_timestamps)

      # Calculate statistics for each value column
      value_stats =
        Enum.reduce(value_columns, %{}, fn column, acc ->
          values =
            Enum.map(data, &Map.get(&1, column))
            |> Enum.filter(&is_number/1)

          if Enum.empty?(values) do
            acc
          else
            stats = calculate_statistics(values)
            Map.put(acc, column, stats)
          end
        end)

      # Return time series statistics
      %{
        start_time: DateTime.to_iso8601(first_timestamp),
        end_time: DateTime.to_iso8601(last_timestamp),
        duration_seconds: duration_seconds,
        point_count: length(data),
        time_frequency: Map.get(gaps, :gap_frequency),
        time_gaps: gaps,
        value_statistics: value_stats
      }
    end
  end

  defp calculate_time_series_stats(_, _, _), do: %{}

  # Helper functions

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
end
