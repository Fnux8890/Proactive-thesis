#!/usr/bin/env elixir

# Fix for CSV viewer component template issues
file_path = "lib/web_service_web/components/file_watcher/csv_viewer_component.ex"
content = File.read!(file_path)

# Replace template line with proper variable assignment in the template
# This addresses the warning about accessing variables inside LiveView templates
fixed_content = content
  # Add @ prefix to any unprefixed variables that should be assigns
  |> String.replace(~r/\bindex\b(?!\w)/, "@index")

# Write the fixed content back
File.write!(file_path, fixed_content)

IO.puts("Fixed template variable access issues in #{file_path}")

# Also fix unused functions in file_watcher_live.ex
file_path2 = "lib/web_service_web/live/file_watcher_live.ex"
content2 = File.read!(file_path2)

# Mark unused variables with underscore
fixed_content2 = content2
  |> String.replace("defp format_content(content, _type, path) do", "defp format_content(_content, _type, path) do")

# Write the fixed content back
File.write!(file_path2, fixed_content2)

IO.puts("Fixed unused variables in #{file_path2}")
