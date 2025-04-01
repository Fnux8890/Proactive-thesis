defmodule WebServiceWeb.PipelineController do
  use WebServiceWeb, :controller

  @data_dir "/data"
  @results_dir "/app/results"

  def index(conn, _params) do
    # Get list of available data files
    data_files = list_data_files()
    
    # Get list of processed results
    results = list_results()

    render(conn, :index, data_files: data_files, results: results)
  end

  def process(conn, %{"file" => file_path}) do
    # Execute the data processing pipeline as a background process
    # This would use Elixir's System.cmd to execute the Docker command
    case System.cmd("docker-compose", ["run", "--rm", "elixir_data_processor", file_path], cd: "/app") do
      {_output, 0} ->
        conn
        |> put_flash(:info, "Data processing started for #{file_path}")
        |> redirect(to: ~p"/pipeline")
      
      {error, _} ->
        conn
        |> put_flash(:error, "Failed to start data processing: #{error}")
        |> redirect(to: ~p"/pipeline")
    end
  end

  def view_result(conn, %{"file" => result_file}) do
    # Read the content of the result file
    case File.read(Path.join(@results_dir, result_file)) do
      {:ok, content} ->
        json = Jason.decode!(content)
        render(conn, :view_result, result: json, file: result_file)
      
      {:error, reason} ->
        conn
        |> put_flash(:error, "Failed to read result file: #{reason}")
        |> redirect(to: ~p"/pipeline")
    end
  end

  def health(conn, _params) do
    json(conn, %{status: "ok"})
  end

  # Helper functions
  defp list_data_files do
    case File.ls(@data_dir) do
      {:ok, files} -> 
        files
        |> Enum.filter(fn file -> String.ends_with?(file, ".csv") end)
        |> Enum.flat_map(fn dir ->
          case File.dir?("#{@data_dir}/#{dir}") do
            true -> 
              case File.ls("#{@data_dir}/#{dir}") do
                {:ok, sub_files} -> 
                  sub_files
                  |> Enum.filter(fn file -> String.ends_with?(file, ".csv") end)
                  |> Enum.map(fn file -> "#{dir}/#{file}" end)
                _ -> []
              end
            false -> [dir]
          end
        end)
      
      {:error, _} -> []
    end
  end

  defp list_results do
    case File.ls(@results_dir) do
      {:ok, files} -> 
        files
        |> Enum.filter(fn file -> String.ends_with?(file, ".json") end)
        |> Enum.sort_by(fn file -> 
          case File.stat(Path.join(@results_dir, file)) do
            {:ok, %{mtime: mtime}} -> mtime
            _ -> {0, 0, 0}
          end
        end, :desc)
      
      {:error, _} -> []
    end
  end
end
