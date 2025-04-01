#!/usr/bin/env elixir

# Fix for CSV viewer component pattern matching with row index
file_path = "lib/web_service_web/components/file_watcher/csv_viewer_component.ex"
content = File.read!(file_path)

# Fix the pattern matching issue in the row for comprehension
fixed_content = String.replace(
  content, 
  "<%= for {cell, @index} <- Enum.with_index(row) do %>",
  "<%= for {cell, _cell_index} <- Enum.with_index(row) do %>"
)

# Write the fixed content back
File.write!(file_path, fixed_content)

IO.puts("Fixed row pattern matching issue in #{file_path}")
