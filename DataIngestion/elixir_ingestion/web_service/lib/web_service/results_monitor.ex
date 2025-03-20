defmodule WebService.ResultsMonitor do
  @moduledoc """
  A GenServer that monitors data processing results and notifies subscribers
  """
  use GenServer
  require Logger
  
  # Client API
  
  def start_link(_) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end
  
  @doc """
  Register a new file processing job
  """
  def register_job(file_path) do
    GenServer.call(__MODULE__, {:register, file_path})
  end
  
  @doc """
  Update the status of a job
  """
  def update_status(job_id, status, details \\ nil) do
    GenServer.cast(__MODULE__, {:update, job_id, status, details})
  end
  
  @doc """
  Get the current status of all jobs
  """
  def get_all_jobs do
    GenServer.call(__MODULE__, :get_all)
  end
  
  @doc """
  Get the status of a specific job
  """
  def get_job(job_id) do
    GenServer.call(__MODULE__, {:get, job_id})
  end
  
  @doc """
  Load results from the results directory
  """
  def load_results do
    GenServer.call(__MODULE__, :load_results)
  end
  
  @doc """
  Get all processing results
  """
  def get_results do
    GenServer.call(__MODULE__, :get_results)
  end
  
  # Server callbacks
  
  @impl true
  def init(_) do
    # Initial state with empty jobs map
    {:ok, %{jobs: %{}, next_id: 1}}
  end
  
  @impl true
  def handle_call({:register, file_path}, _from, state) do
    # Create a new job with pending status
    job_id = state.next_id
    job = %{
      id: job_id,
      file_path: file_path,
      status: :pending,
      started_at: DateTime.utc_now(),
      details: nil
    }
    
    # Update state with new job
    new_state = %{
      state | 
      jobs: Map.put(state.jobs, job_id, job),
      next_id: job_id + 1
    }
    
    # Broadcast job update
    Phoenix.PubSub.broadcast(WebService.PubSub, "jobs", {:job_updated, job})
    
    {:reply, {:ok, job_id}, new_state}
  end
  
  @impl true
  def handle_call(:get_all, _from, state) do
    {:reply, Map.values(state.jobs), state}
  end
  
  @impl true
  def handle_call({:get, job_id}, _from, state) do
    job = Map.get(state.jobs, job_id)
    {:reply, job, state}
  end
  
  @impl true
  def handle_call(:load_results, _from, state) do
    # Look for results in the results directory
    results_dir = "../results"
    results = if File.exists?(results_dir) do
      results_dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, "_analysis.json"))
      |> Enum.map(fn filename ->
        file_path = Path.join(results_dir, filename)
        
        try do
          json = File.read!(file_path)
          {:ok, data} = Jason.decode(json)
          
          # Create a job entry from the results file
          %{
            id: System.unique_integer([:positive]),
            file_path: get_in(data, ["data", "file_path"]) || "Unknown",
            status: :completed,
            started_at: nil,
            completed_at: data["timestamp"],
            details: data
          }
        rescue
          _ -> nil
        end
      end)
      |> Enum.reject(&is_nil/1)
    else
      []
    end
    
    # Add the existing results as jobs
    jobs = results
    |> Enum.reduce(state.jobs, fn job, acc ->
      Map.put(acc, job.id, job)
    end)
    
    # Update state with loaded results
    new_state = %{state | jobs: jobs}
    
    # Broadcast loaded jobs
    Enum.each(results, fn job ->
      Phoenix.PubSub.broadcast(WebService.PubSub, "jobs", {:job_updated, job})
    end)
    
    {:reply, {:ok, length(results)}, new_state}
  end
  
  @impl true
  def handle_call(:get_results, _from, state) do
    results = state.jobs
    |> Map.values()
    |> Enum.filter(fn job -> job.status == :completed end)
    |> Enum.map(fn job -> job.details end)
    
    {:reply, results, state}
  end
  
  @impl true
  def handle_cast({:update, job_id, status, details}, state) do
    # Find the job
    case Map.get(state.jobs, job_id) do
      nil ->
        # Job not found
        Logger.warning("Attempted to update non-existent job #{job_id}")
        {:noreply, state}
        
      job ->
        # Update the job
        updated_job = %{
          job |
          status: status,
          details: details || job.details
        }
        
        # Add completed_at timestamp if the job is completed
        updated_job = if status in [:completed, :failed] do
          Map.put(updated_job, :completed_at, DateTime.utc_now())
        else
          updated_job
        end
        
        # Update state
        new_state = %{
          state |
          jobs: Map.put(state.jobs, job_id, updated_job)
        }
        
        # Broadcast job update
        Phoenix.PubSub.broadcast(WebService.PubSub, "jobs", {:job_updated, updated_job})
        
        {:noreply, new_state}
    end
  end
end
