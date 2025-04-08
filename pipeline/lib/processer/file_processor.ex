defmodule Processer.FileProcessor do
  use GenServer
  require Logger # Ensure Logger is required

  # Add a prominent log message to track module loading
  Logger.warn("LOADING FILE PROCESSOR MODULE - VERSION: #{inspect(Application.spec(:pipeline, :vsn))} - TIMESTAMP: #{inspect(DateTime.utc_now())}")

  defp parse_csv_content(content) do
    try do
      # Split content into lines
      lines = String.split(content, "\n", trim: true)

      case lines do
        # Ensure there's at least one line (header) and potentially data lines
        [_header_line | data_lines] when is_list(data_lines) ->
          # Join the data lines back into a single string for parsing
          data_lines_string = Enum.join(data_lines, "\n")

          # Define an ad-hoc parser with the correct options
          MyParser = NimbleCSV.define(parse_string(), separator: ";", headers: false)

          # Parse only the data lines using the defined parser
          parsed_data = MyParser.parse_string(data_lines_string)

          # Log success for debugging (temporary)
          Logger.debug("Successfully parsed data lines for CSV.")

          # TODO: Re-integrate header cleaning and map creation later
          # For now, just return the list of lists if successful
          {:ok, parsed_data}

        # Handle cases with empty or only header line
        [] ->
          Logger.warning("CSV file is empty.")
          {:error, :empty_file}

        [_header_line] ->
           Logger.warning("CSV file contains only a header line.")
           {:ok, []} # Or consider {:error, :no_data_lines}
      end # End of case block
    rescue
      e in _ -> # Catch any exception
      # Handle any exceptions that might occur during parsing
      Logger.error("Error parsing CSV content: #{inspect(e)}")
      {:error, :parse_error}
    end
  end
end
