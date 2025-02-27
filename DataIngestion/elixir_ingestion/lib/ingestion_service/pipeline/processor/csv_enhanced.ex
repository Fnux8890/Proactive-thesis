defmodule IngestionService.Pipeline.Processor.CSVEnhanced do
  require Logger

  @moduledoc """
  Enhanced CSV processing module that provides automatic format detection
  and intelligent handling of different CSV variants.
  """

  @doc """
  Processes a CSV file with automatic format detection and intelligent parsing.

  ## Options

  * `:sample_size` - Number of lines to sample for format detection (default: 1000)
  * `:separator` - Force a specific separator character (overrides detection)
  * `:has_headers` - Force whether the file has headers (overrides detection)
  * `:encoding` - Force a specific encoding (overrides detection)
  * `:timestamp_formats` - List of custom timestamp formats to look for
  * `:trim_cells` - Whether to trim whitespace from cells (default: true)
  * `:skip_lines` - Number of lines to skip at the beginning (default: 0)
  * `:max_rows` - Maximum number of rows to process (default: all)
  """
  def process_csv(file_path, options \\ []) do
    # Start time for performance monitoring
    start_time = System.monotonic_time(:millisecond)

    # Read a sample of the file
    sample_size = Keyword.get(options, :sample_size, 1000)
    sample = read_sample(file_path, sample_size)

    # Automatically detect format parameters
    detected = detect_format_parameters(sample, options)

    # Merge detected options with user options, with user options taking precedence
    parsing_options = Map.merge(detected, Map.new(options))

    # Parse the file with the enhanced options
    result = parse_csv_with_options(file_path, parsing_options)

    # Calculate processing time
    end_time = System.monotonic_time(:millisecond)
    processing_time = end_time - start_time

    # Log performance metrics
    Logger.debug(fn ->
      "CSV processing completed in #{processing_time}ms - File: #{file_path}, Rows: #{length(result.data)}"
    end)

    # Return processed data with metadata
    Map.put(result, :processing_time_ms, processing_time)
  end

  @doc """
  Detects CSV format parameters from a sample of the file.
  """
  def detect_format_parameters(sample, options \\ []) do
    # Extract the sample lines
    sample_lines = String.split(sample, ~r/\r?\n/)

    # Filter out empty lines
    non_empty_lines = Enum.reject(sample_lines, &(String.trim(&1) == ""))

    # If we have no lines, we can't detect anything
    if Enum.empty?(non_empty_lines) do
      %{
        separator: ",",
        has_headers: true,
        encoding: "UTF-8",
        timestamp_formats: [],
        trim_cells: true,
        skip_lines: 0
      }
    else
      # Detect parameters from the sample
      %{
        separator: detect_separator(non_empty_lines, options),
        has_headers: detect_headers(non_empty_lines, options),
        encoding: detect_encoding(sample, options),
        timestamp_formats: detect_timestamp_formats(non_empty_lines, options),
        trim_cells: Keyword.get(options, :trim_cells, true),
        skip_lines: Keyword.get(options, :skip_lines, 0)
      }
    end
  end

  @doc """
  Reads a sample of lines from a file for format detection.
  """
  def read_sample(file_path, sample_size) do
    # Ensure we have a valid sample size
    actual_sample_size = max(10, sample_size)

    try do
      # Open the file
      {:ok, file} = File.open(file_path, [:read, :utf8])

      # Read the sample lines
      sample =
        1..actual_sample_size
        |> Enum.reduce_while("", fn _, acc ->
          case IO.read(file, :line) do
            :eof -> {:halt, acc}
            line -> {:cont, acc <> line}
          end
        end)

      # Close the file
      File.close(file)

      sample
    rescue
      e ->
        Logger.error("Error reading sample from #{file_path}: #{inspect(e)}")
        ""
    end
  end

  @doc """
  Detects the most likely separator character used in the CSV.
  """
  def detect_separator(lines, options \\ []) do
    # If a separator is provided in options, use it
    case Keyword.get(options, :separator) do
      nil ->
        # Try to automatically detect
        do_detect_separator(lines)

      separator ->
        # Use the provided separator
        separator
    end
  end

  # Implements the actual separator detection logic
  defp do_detect_separator(lines) do
    # Take a subset of lines for detection
    sample_lines = Enum.take(lines, min(10, length(lines)))

    # Common separators to check
    separators = [",", ";", "\t", "|"]

    # Count fields for each potential separator
    separator_scores =
      Enum.map(separators, fn sep ->
        # Calculate consistency score for this separator
        field_counts =
          Enum.map(sample_lines, fn line ->
            line |> String.split(sep) |> length()
          end)

        # Calculate mean field count
        mean_count =
          if Enum.empty?(field_counts) do
            0
          else
            Enum.sum(field_counts) / length(field_counts)
          end

        # Calculate variance (how consistent the field counts are)
        variance =
          if Enum.empty?(field_counts) or length(field_counts) == 1 do
            0
          else
            field_counts
            |> Enum.map(fn count -> :math.pow(count - mean_count, 2) end)
            |> Enum.sum()
            |> Kernel./(length(field_counts))
          end

        # Calculate the number of fields (higher is better, as it likely means proper splitting)
        field_score = mean_count

        # Calculate consistency score (lower variance is better)
        consistency_score = 1 / (1 + variance)

        # Check for quote consistency (if field contains quotes, separator is likely correct)
        quote_score = quote_consistency_score(sample_lines, sep)

        # Calculate field balance score (are fields roughly similar length?)
        balance_score = field_balance_score(sample_lines, sep)

        # Weighted combined score
        score =
          field_score * 0.4 + consistency_score * 0.3 + quote_score * 0.2 + balance_score * 0.1

        {sep, score}
      end)

    # Select the separator with the highest score
    {best_separator, _} =
      separator_scores
      |> Enum.sort_by(fn {_, score} -> score end, :desc)
      |> List.first()

    best_separator
  end

  # Calculates score based on proper quote handling
  defp quote_consistency_score(lines, separator) do
    # Count balanced quotes in each line
    quote_scores =
      Enum.map(lines, fn line ->
        fields = String.split(line, separator)

        # Count balanced quotes in each field
        balanced_count =
          Enum.count(fields, fn field ->
            # Count quotes
            quote_count = field |> String.codepoints() |> Enum.count(&(&1 == "\""))

            # Check if balanced
            rem(quote_count, 2) == 0
          end)

        # Calculate score for this line
        if length(fields) == 0 do
          0
        else
          balanced_count / length(fields)
        end
      end)

    # Average the scores
    if Enum.empty?(quote_scores) do
      0
    else
      Enum.sum(quote_scores) / length(quote_scores)
    end
  end

  # Calculates score based on field length balance
  defp field_balance_score(lines, separator) do
    # Calculate field length variance for each line
    field_variances =
      Enum.map(lines, fn line ->
        fields = String.split(line, separator)

        # Calculate field lengths
        field_lengths = Enum.map(fields, &String.length/1)

        # Calculate mean length
        mean_length =
          if Enum.empty?(field_lengths) do
            0
          else
            Enum.sum(field_lengths) / length(field_lengths)
          end

        # Calculate variance
        if Enum.empty?(field_lengths) or length(field_lengths) == 1 do
          0
        else
          variance =
            field_lengths
            |> Enum.map(fn length -> :math.pow(length - mean_length, 2) end)
            |> Enum.sum()
            |> Kernel./(length(field_lengths))

          # Normalize variance by mean length (to handle different line lengths)
          if mean_length == 0 do
            0
          else
            variance / mean_length
          end
        end
      end)

    # Average the variances and convert to a score (lower variance is better)
    if Enum.empty?(field_variances) do
      0
    else
      avg_variance = Enum.sum(field_variances) / length(field_variances)
      1 / (1 + avg_variance)
    end
  end

  @doc """
  Detects whether the CSV file has a header row.
  """
  def detect_headers(lines, options \\ []) do
    # If header presence is specified in options, use it
    case Keyword.get(options, :has_headers) do
      nil ->
        # Try to automatically detect
        do_detect_headers(lines, detect_separator(lines, options))

      has_headers ->
        # Use the provided setting
        has_headers
    end
  end

  # Implements the actual header detection logic
  defp do_detect_headers(lines, separator) do
    # We need at least 2 lines to detect headers
    if length(lines) < 2 do
      # Default to true if we can't detect
      true
    else
      # Get the first two lines
      [first_line | rest] = lines
      second_line = List.first(rest)

      # Split into fields
      first_fields = String.split(first_line, separator)
      second_fields = String.split(second_line, separator)

      # Check if the number of fields matches
      if length(first_fields) != length(second_fields) do
        # Different field counts suggest first line is not a header
        false
      else
        # Check if first line looks like headers:
        # 1. Headers often have different data types than data rows
        # 2. Headers are often shorter
        # 3. Headers rarely have numbers only
        # 4. Headers rarely have the same values as data rows

        # Check data type difference
        types_different =
          Enum.zip(first_fields, second_fields)
          |> Enum.count(fn {first, second} ->
            looks_like_number?(first) != looks_like_number?(second)
          end)
          |> Kernel./(length(first_fields))
          |> Kernel.>(0.5)

        # Check if first line fields are shorter on average
        first_avg_length =
          first_fields
          |> Enum.map(&String.length/1)
          |> Enum.sum()
          |> Kernel./(length(first_fields))

        second_avg_length =
          second_fields
          |> Enum.map(&String.length/1)
          |> Enum.sum()
          |> Kernel./(length(second_fields))

        shorter_fields = first_avg_length < second_avg_length

        # Check if first line has fewer numbers
        first_number_ratio =
          first_fields
          |> Enum.count(&looks_like_number?/1)
          |> Kernel./(length(first_fields))

        second_number_ratio =
          second_fields
          |> Enum.count(&looks_like_number?/1)
          |> Kernel./(length(second_fields))

        fewer_numbers = first_number_ratio < second_number_ratio

        # Calculate a header probability score
        header_score =
          if(types_different, do: 0.5, else: 0) +
            if(shorter_fields, do: 0.3, else: 0) +
            if fewer_numbers, do: 0.2, else: 0

        # Determine if it looks like a header
        header_score > 0.5
      end
    end
  end

  # Checks if a string looks like a number
  defp looks_like_number?(str) do
    case Float.parse(String.trim(str)) do
      {_, ""} -> true
      _ -> false
    end
  end

  @doc """
  Detects the encoding of the file content.
  """
  def detect_encoding(sample, options \\ []) do
    # If encoding is specified in options, use it
    case Keyword.get(options, :encoding) do
      nil ->
        # Try to automatically detect
        do_detect_encoding(sample)

      encoding ->
        # Use the provided encoding
        encoding
    end
  end

  # Implements the actual encoding detection logic
  defp do_detect_encoding(sample) do
    # This is a simplified version of encoding detection
    # A more robust solution would use libraries like 'chardet'

    # Check for common UTF-8 patterns
    if is_valid_utf8?(sample) do
      "UTF-8"
    else
      # Check for UTF-16 BOM markers
      cond do
        String.starts_with?(sample, <<0xFF, 0xFE>>) -> "UTF-16LE"
        String.starts_with?(sample, <<0xFE, 0xFF>>) -> "UTF-16BE"
        # Default to Latin-1 which can read most Western text files
        true -> "ISO-8859-1"
      end
    end
  end

  # Checks if a binary is valid UTF-8
  defp is_valid_utf8?(binary) do
    case String.valid?(binary) do
      true -> true
      false -> false
    end
  end

  @doc """
  Detects timestamp formats in the CSV data.
  """
  def detect_timestamp_formats(lines, options \\ []) do
    # If timestamp formats are specified in options, use them
    case Keyword.get(options, :timestamp_formats) do
      nil ->
        # Try to automatically detect
        do_detect_timestamp_formats(lines, detect_separator(lines, options))

      formats ->
        # Use the provided formats
        formats
    end
  end

  # Implements the actual timestamp format detection logic
  defp do_detect_timestamp_formats(lines, separator) do
    # Common timestamp format patterns
    timestamp_patterns = [
      {~r/^\d{4}-\d{2}-\d{2}$/, "yyyy-MM-dd"},
      {~r/^\d{2}\/\d{2}\/\d{4}$/, "MM/dd/yyyy"},
      {~r/^\d{4}\/\d{2}\/\d{2}$/, "yyyy/MM/dd"},
      {~r/^\d{2}-\d{2}-\d{4}$/, "MM-dd-yyyy"},
      {~r/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/, "ISO8601"},
      {~r/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/, "yyyy-MM-dd HH:mm:ss"},
      {~r/^\d{2}\/\d{2}\/\d{4} \d{2}:\d{2}/, "MM/dd/yyyy HH:mm"},
      {~r/^\d{10}$/, "Unix timestamp (seconds)"},
      {~r/^\d{13}$/, "Unix timestamp (milliseconds)"}
    ]

    # Extract all fields from the lines
    all_fields =
      Enum.flat_map(lines, fn line ->
        String.split(line, separator)
      end)
      |> Enum.map(&String.trim/1)

    # Check each timestamp pattern
    format_matches =
      Enum.reduce(timestamp_patterns, %{}, fn {pattern, format}, acc ->
        # Count matches for this pattern
        matches = Enum.count(all_fields, &String.match?(&1, pattern))

        # Only keep formats with significant matches
        if matches > 0 do
          Map.put(acc, format, matches)
        else
          acc
        end
      end)

    # Return detected formats sorted by frequency
    format_matches
    |> Enum.sort_by(fn {_format, count} -> count end, :desc)
    |> Enum.map(fn {format, _count} -> format end)
  end

  @doc """
  Parses a CSV file with the detected options.
  """
  def parse_csv_with_options(file_path, options) do
    # Get options with defaults
    separator = Map.get(options, :separator, ",")
    has_headers = Map.get(options, :has_headers, true)
    encoding = Map.get(options, :encoding, "UTF-8")
    trim_cells = Map.get(options, :trim_cells, true)
    skip_lines = Map.get(options, :skip_lines, 0)
    max_rows = Map.get(options, :max_rows, :infinity)

    # Create parser options
    parser_options = [
      separator: separator,
      skip_lines: skip_lines,
      max_rows: max_rows,
      trim_cells: trim_cells
    ]

    try do
      # Read and parse the file
      {:ok, file} = File.open(file_path, [:read, encoding: String.to_atom(encoding)])

      # Parse the file line by line
      {headers, rows} = parse_csv_file(file, parser_options, has_headers)

      # Close the file
      File.close(file)

      # Return the parsed data with metadata
      %{
        data: rows,
        headers: headers,
        format: %{
          separator: separator,
          has_headers: has_headers,
          encoding: encoding,
          timestamp_formats: Map.get(options, :timestamp_formats, [])
        },
        row_count: length(rows),
        file_path: file_path
      }
    rescue
      e ->
        Logger.error("Error parsing CSV file #{file_path}: #{inspect(e)}")

        # Return error data
        %{
          data: [],
          headers: [],
          format: %{
            separator: separator,
            has_headers: has_headers,
            encoding: encoding
          },
          row_count: 0,
          file_path: file_path,
          error: "#{inspect(e)}"
        }
    end
  end

  # Parses a CSV file line by line
  defp parse_csv_file(file, options, has_headers) do
    # Skip initial lines if needed
    skip_count = Keyword.get(options, :skip_lines, 0)

    if skip_count > 0 do
      1..skip_count
      |> Enum.each(fn _ -> IO.read(file, :line) end)
    end

    # Parse headers if present
    headers =
      if has_headers do
        case IO.read(file, :line) do
          :eof -> []
          header_line -> parse_csv_line(header_line, options)
        end
      else
        []
      end

    # Parse rows
    max_rows = Keyword.get(options, :max_rows, :infinity)
    rows = parse_csv_rows(file, options, [], 0, max_rows)

    # Return headers and rows
    {headers, rows}
  end

  # Parses CSV rows up to max_rows
  defp parse_csv_rows(file, options, acc, count, max_rows) when count >= max_rows do
    Enum.reverse(acc)
  end

  defp parse_csv_rows(file, options, acc, count, max_rows) do
    case IO.read(file, :line) do
      :eof ->
        Enum.reverse(acc)

      line ->
        parsed_line = parse_csv_line(line, options)
        parse_csv_rows(file, options, [parsed_line | acc], count + 1, max_rows)
    end
  end

  # Parses a single CSV line considering quotes and separators
  defp parse_csv_line(line, options) do
    separator = Keyword.get(options, :separator, ",")
    trim_cells = Keyword.get(options, :trim_cells, true)

    # Use regex to handle quotes properly
    fields = extract_csv_fields(line, separator)

    # Trim if needed
    if trim_cells do
      Enum.map(fields, &String.trim/1)
    else
      fields
    end
  end

  # Extracts fields from a CSV line respecting quotes
  defp extract_csv_fields(line, separator) do
    # This regex-based approach handles quotes properly
    # It matches either:
    # 1. Quoted field: starts and ends with quotes, can contain any character including separators
    # 2. Unquoted field: any sequence of characters not containing the separator

    # Escape the separator for use in regex
    escaped_separator = Regex.escape(separator)

    # The regex pattern for fields
    pattern = ~r/(\"[^\"]*(?:\"\"[^\"]*)*\"|[^#{escaped_separator}]*)#{escaped_separator}?/

    # Extract all matches
    Regex.scan(pattern, String.trim(line) <> separator, capture: :all_but_first)
    |> Enum.map(fn [field] ->
      clean_csv_field(field)
    end)
  end

  # Cleans a CSV field: removes quotes and converts "" to "
  defp clean_csv_field(field) do
    if String.starts_with?(field, "\"") && String.ends_with?(field, "\"") do
      # Remove surrounding quotes
      field
      |> String.slice(1..-2)
      # Convert escaped quotes
      |> String.replace("\"\"", "\"")
    else
      field
    end
  end
end
