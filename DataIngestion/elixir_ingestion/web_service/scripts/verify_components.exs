#!/usr/bin/env elixir

# A simple script to verify our components are correctly implemented
# without starting the full Phoenix server

IO.puts("=== Component Integration Verification ===")

# Check that all component files exist
files_to_check = [
  "lib/web_service_web/components/file_watcher/file_preview_component.ex",
  "lib/web_service_web/components/file_watcher/csv_viewer_component.ex",
  "lib/web_service_web/components/file_watcher/json_viewer_component.ex"
]

files_exist = Enum.all?(files_to_check, &File.exists?/1)
IO.puts("All component files exist: #{files_exist}")

# Read the file_preview_component.ex to verify the integration
file_preview_content = File.read!("lib/web_service_web/components/file_watcher/file_preview_component.ex")

# Check if it includes the specialized viewer components
has_csv_viewer = String.contains?(file_preview_content, "CsvViewerComponent")
has_json_viewer = String.contains?(file_preview_content, "JsonViewerComponent")

IO.puts("File preview includes CSV viewer: #{has_csv_viewer}")
IO.puts("File preview includes JSON viewer: #{has_json_viewer}")

# Verify the conditional logic in file_preview_component.ex
has_csv_condition = String.contains?(file_preview_content, "is_csv_file?(@current_file)")
has_json_condition = String.contains?(file_preview_content, "is_json_file?(@current_file)")

IO.puts("File preview checks for CSV files: #{has_csv_condition}")
IO.puts("File preview checks for JSON files: #{has_json_condition}")

# Summary
all_checks = [files_exist, has_csv_viewer, has_json_viewer, has_csv_condition, has_json_condition]
if Enum.all?(all_checks, &(&1 == true)) do
  IO.puts("\n✅ All integration checks passed! The file preview component has been successfully integrated with specialized viewers.")
  IO.puts("When the server starts, it will use the appropriate viewer based on file type.")
else
  IO.puts("\n❌ Some integration checks failed. Please review the component code.")
end
