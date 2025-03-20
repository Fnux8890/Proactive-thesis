#!/usr/bin/env elixir

# Simple fix for unused variables in file_watcher_live.ex
file_path = "lib/web_service_web/live/file_watcher_live.ex"
content = File.read!(file_path)

# Replace the unused variables with prefixed versions
fixed_content = String.replace(
  content, 
  "def handle_info({:load_data, folder, path, file}, socket) do", 
  "def handle_info({:load_data, _folder, _path, _file}, socket) do"
)

# Write the fixed content back
File.write!(file_path, fixed_content)

IO.puts("Fixed unused variables in #{file_path}")
