#!/usr/bin/env elixir
# Simple data file processor
# This script processes a single data file and outputs basic analysis

# Add Jason dependency
Mix.install([
  {:jason, "~> 1.4"}
])

defmodule DataProcessor do
  @moduledoc """
  Module for processing data files (CSV, JSON, etc.) with minimal dependencies.
  """

  @doc """
  Process a data file and return analysis results
  """
  def process_file(file_path) do
    IO.puts("Processing file: #{file_path}")
    
    # Determine file type from extension
    file_type = file_path |> Path.extname() |> String.downcase()
    
    # Read file
    case File.read(file_path) do
      {:ok, content} ->
        # Process based on file type
        result = case file_type do
          ".csv" -> process_csv(content, file_path)
          ".json" -> process_json(content, file_path)
          _ -> {:error, "Unsupported file type: #{file_type}"}
        end
        
        # Save results
        save_results(result, file_path)
        
      {:error, reason} ->
        IO.puts("Error reading file: #{inspect(reason)}")
        {:error, "File read error: #{inspect(reason)}"}
    end
  end
  
  @doc """
  Process CSV content
  """
  def process_csv(content, file_path) do
    IO.puts("Processing CSV file...")
    
    # Split content into lines
    lines = String.split(content, "\n", trim: true)
    
    if Enum.empty?(lines) do
      {:error, "Empty CSV file"}
    else
      # Check for delimiter (comma or semicolon)
      first_line = Enum.at(lines, 0)
      delimiter = if String.contains?(first_line, ";"), do: ";", else: ","
      
      # Parse header row
      headers = first_line
        |> String.split(delimiter, trim: true)
        |> Enum.map(&String.trim/1)
      
      # Parse data rows (limited sample for now)
      sample_size = min(10, length(lines) - 1)
      sample_rows = lines
        |> Enum.slice(1, sample_size)
        |> Enum.map(fn line ->
          values = String.split(line, delimiter, trim: true)
          if length(values) == length(headers) do
            Enum.zip(headers, values) |> Enum.into(%{})
          else
            %{"_error" => "Column count mismatch", "_line" => line}
          end
        end)
      
      # Build basic stats
      stats = %{
        file_path: file_path,
        file_size_bytes: byte_size(content),
        line_count: length(lines),
        header_count: length(headers),
        headers: headers,
        delimiter: delimiter,
        sample_size: sample_size,
        sample: sample_rows
      }
      
      {:ok, stats}
    end
  end
  
  @doc """
  Process JSON content
  """
  def process_json(content, file_path) do
    IO.puts("Processing JSON file...")
    
    try do
      # Parse JSON using Jason
      json_data = Jason.decode!(content)
      
      # Determine JSON structure
      structure = cond do
        is_list(json_data) -> 
          %{
            type: "array",
            count: length(json_data),
            sample: Enum.take(json_data, 5)
          }
        is_map(json_data) -> 
          %{
            type: "object",
            keys: Map.keys(json_data),
            sample: json_data
          }
        true -> 
          %{
            type: "primitive",
            value: json_data
          }
      end
      
      # Build stats
      stats = %{
        file_path: file_path,
        file_size_bytes: byte_size(content),
        structure: structure
      }
      
      {:ok, stats}
    rescue
      e -> 
        IO.puts("Error parsing JSON: #{inspect(e)}")
        {:error, "JSON parse error: #{inspect(e)}"}
    end
  end
  
  @doc """
  Save the processing results
  """
  def save_results(result, original_file_path) do
    # Create results directory
    results_dir = "results"
    File.mkdir_p!(results_dir)
    
    # Generate output filename
    basename = Path.basename(original_file_path)
    output_file = Path.join(results_dir, "#{basename}_analysis.json")
    
    # Convert result to JSON and write to file
    {status, data} = result
    result_json = %{
      status: status,
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      data: data
    }
    
    # Convert to JSON and write to file
    json_string = Jason.encode!(result_json, pretty: true)
    
    # Write to file
    File.write!(output_file, json_string)
    
    IO.puts("Results saved to: #{output_file}")
    
    # Return the final result
    %{result: result, output_file: output_file}
  end
end

# Entry point for script execution
if System.argv() |> length() > 0 do
  [file_path | _] = System.argv()
  DataProcessor.process_file(file_path)
else
  IO.puts("""
  Usage: elixir process_data_file.exs <file_path>
  
  Example: elixir process_data_file.exs /data/sample_data.csv
  """)
end
