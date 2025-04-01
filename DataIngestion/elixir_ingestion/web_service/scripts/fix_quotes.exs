#!/usr/bin/env elixir

defmodule QuoteFixer do
  @target_dir "lib/web_service_web/components/file_watcher"
  @file_pattern "*.ex"

  def run do
    Path.wildcard(Path.join(@target_dir, @file_pattern))
    |> Enum.each(&fix_quotes_in_file/1)
  end

  defp fix_quotes_in_file(file_path) do
    IO.puts("Processing file: #{file_path}")
    
    content = File.read!(file_path)
    fixed_content = fix_quoted_strings(content)
    
    if content != fixed_content do
      File.write!(file_path, fixed_content)
      IO.puts("  Fixed quotes in #{file_path}")
    else
      IO.puts("  No changes needed in #{file_path}")
    end
  end

  defp fix_quoted_strings(content) do
    # Fix all single quotes in class attributes
    content
    |> String.replace(~r/(do: )'([^']+)'/, "\\1\"\\2\"")
    |> String.replace(~r/(else: )'([^']+)'/, "\\1\"\\2\"")
  end
end

QuoteFixer.run()
