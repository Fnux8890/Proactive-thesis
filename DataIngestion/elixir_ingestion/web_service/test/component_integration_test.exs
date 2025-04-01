#!/usr/bin/env elixir

defmodule ComponentIntegrationTest do
  @moduledoc """
  A simple test script to verify integration of specialized CSV and JSON viewers
  in the FilePreviewComponent.

  This test can be run independently without starting the full Phoenix server.
  """

  # Import required modules
  Code.require_file("lib/web_service_web/components/file_watcher/file_preview_component.ex")
  Code.require_file("lib/web_service_web/components/file_watcher/csv_viewer_component.ex")
  Code.require_file("lib/web_service_web/components/file_watcher/json_viewer_component.ex")

  # Test the file type detection functions
  def test_file_type_detection do
    alias WebServiceWeb.Components.FileWatcher.FilePreviewComponent

    # Test CSV detection
    IO.puts("Testing CSV file detection:")
    csv_result = FilePreviewComponent.is_csv_file?("data.csv")
    IO.puts("  'data.csv' is CSV? #{csv_result}")
    
    # Test JSON detection
    IO.puts("Testing JSON file detection:")
    json_result = FilePreviewComponent.is_json_file?("data.json")
    IO.puts("  'data.json' is JSON? #{json_result}")
  end

  # Run all tests
  def run do
    IO.puts("Running component integration tests...\n")
    test_file_type_detection()
    IO.puts("\nTests completed.")
  end
end

# Execute the tests
ComponentIntegrationTest.run()
