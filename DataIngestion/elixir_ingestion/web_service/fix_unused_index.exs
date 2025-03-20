#!/usr/bin/env elixir

# Fix for CSV viewer component unused index variable
file_path = "lib/web_service_web/components/file_watcher/csv_viewer_component.ex"
content = File.read!(file_path)

# Fix the phx-value-column to use the index variable instead of @index
fixed_content = String.replace(
  content, 
  "phx-value-column={@index}",
  "phx-value-column={index}"
)

# Write the fixed content back
File.write!(file_path, fixed_content)

IO.puts("Fixed unused index variable in #{file_path}")
