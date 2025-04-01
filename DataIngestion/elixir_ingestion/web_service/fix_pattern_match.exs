#!/usr/bin/env elixir

# Fix for CSV viewer component pattern matching issues
file_path = "lib/web_service_web/components/file_watcher/csv_viewer_component.ex"
content = File.read!(file_path)

# Fix the pattern matching issue in the for comprehension
# We can't use @index on the left side of a pattern match
fixed_content = String.replace(
  content, 
  "<%= for {header, @index} <- Enum.with_index(@headers) do %>",
  "<%= for {header, index} <- Enum.with_index(@headers) do %>"
)

# Write the fixed content back
File.write!(file_path, fixed_content)

IO.puts("Fixed pattern matching issue in #{file_path}")
