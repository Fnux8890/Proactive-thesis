defmodule WebServiceWeb.ApiController do
  use Phoenix.Controller
  alias WebService.ResultsMonitor
  
  @doc """
  Process a file via API request
  Expects "file_path" parameter in the request body
  """
  def process_file(conn, %{"file_path" => file_path}) do
    # Register the job
    {:ok, job_id} = ResultsMonitor.register_job(file_path)
    
    # Start processing the file asynchronously
    Task.start(fn ->
      # Update job status to processing
      ResultsMonitor.update_status(job_id, :processing)
      
      # Construct the command to run our processor
      result = System.cmd("elixir", ["../process_data_file.exs", file_path], cd: File.cwd!())
      
      case result do
        {output, 0} ->
          # Load the results file
          results_dir = "../results"
          basename = Path.basename(file_path)
          output_file = Path.join(results_dir, "#{basename}_analysis.json")
          
          if File.exists?(output_file) do
            # Read results and update job status
            json = File.read!(output_file)
            {:ok, data} = Jason.decode(json)
            ResultsMonitor.update_status(job_id, :completed, data)
          else
            # File processed but no results file found
            ResultsMonitor.update_status(job_id, :failed, %{
              "error" => "Results file not found",
              "output" => output
            })
          end
          
        {output, _} ->
          # Process failed
          ResultsMonitor.update_status(job_id, :failed, %{
            "error" => "Process exited with non-zero status",
            "output" => output
          })
      end
    end)
    
    # Return the job ID
    json(conn, %{job_id: job_id, status: "pending"})
  end
  
  def process_file(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{error: "Missing 'file_path' parameter"})
  end
  
  @doc """
  List all jobs
  """
  def list_jobs(conn, _params) do
    jobs = ResultsMonitor.get_all_jobs()
    |> Enum.map(&format_job_for_json/1)
    
    json(conn, jobs)
  end
  
  @doc """
  Get a specific job by ID
  """
  def get_job(conn, %{"id" => id}) do
    job_id = case Integer.parse(id) do
      {num, ""} -> num
      _ -> nil
    end
    
    case ResultsMonitor.get_job(job_id) do
      nil ->
        conn
        |> put_status(404)
        |> json(%{error: "Job not found"})
        
      job ->
        json(conn, format_job_for_json(job))
    end
  end
  
  # Helper function to format job for JSON response
  defp format_job_for_json(job) do
    # Convert struct or map to a simple map with string keys
    job
    |> Map.from_struct()
    |> Map.new(fn {k, v} -> {to_string(k), v} end)
  rescue
    # If it's already a map, just use it
    _ -> job
  end
end
