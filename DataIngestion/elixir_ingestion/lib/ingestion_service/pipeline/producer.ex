defmodule IngestionService.Pipeline.Producer do
  use GenStage
  require Logger

  @moduledoc """
  A GenStage producer that generates events from file system changes.
  This is the first stage in the ingestion pipeline.
  """

  # Client API

  def start_link(_opts) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  # Push a new file event into the producer
  def notify_file_change(file_path) do
    GenStage.cast(__MODULE__, {:file_change, file_path})
  end

  # Server callbacks

  @impl true
  def init(:ok) do
    Logger.info("Starting ingestion producer")
    # The producer will buffer demand and events
    {:producer, %{events: [], demand: 0}}
  end

  @impl true
  def handle_cast({:file_change, file_path}, %{events: events, demand: 0} = state) do
    # If there's no pending demand, just buffer the event
    Logger.debug("Buffering file change event for #{file_path}")
    {:noreply, [], %{state | events: events ++ [create_event(file_path)]}}
  end

  @impl true
  def handle_cast({:file_change, file_path}, %{events: events, demand: demand} = state) do
    # If there's pending demand, immediately dispatch the event
    event = create_event(file_path)
    Logger.debug("Dispatching file change event for #{file_path}")
    
    # Track this event for telemetry
    :telemetry.execute(
      [:ingestion_service, :pipeline, :produce],
      %{count: 1},
      %{path: file_path}
    )
    
    new_events = events ++ [event]
    {to_dispatch, remaining} = Enum.split(new_events, demand)
    new_demand = demand - length(to_dispatch)
    
    {:noreply, to_dispatch, %{state | events: remaining, demand: new_demand}}
  end

  @impl true
  def handle_demand(demand, %{events: events, demand: pending_demand} = state) when demand > 0 do
    Logger.debug("Received demand for #{demand} events")
    # Dispatch events if any, otherwise store the demand
    new_demand = pending_demand + demand
    {to_dispatch, remaining} = Enum.split(events, new_demand)
    remaining_demand = new_demand - length(to_dispatch)
    
    if length(to_dispatch) > 0 do
      Logger.debug("Dispatching #{length(to_dispatch)} events from buffer")
    end
    
    {:noreply, to_dispatch, %{state | events: remaining, demand: remaining_demand}}
  end

  # Create an event struct with file metadata
  defp create_event(file_path) do
    # Extract file information
    %{
      file_path: file_path,
      timestamp: DateTime.utc_now(),
      file_type: determine_file_type(file_path),
      file_size: get_file_size(file_path),
      source: determine_source(file_path),
      status: :new
    }
  end

  # Determine file type based on extension
  defp determine_file_type(file_path) do
    case Path.extname(file_path) |> String.downcase() do
      ".csv" -> :csv
      ".json" -> :json
      ".xlsx" -> :excel
      ".xls" -> :excel
      _ -> :unknown
    end
  end

  # Get file size
  defp get_file_size(file_path) do
    case File.stat(file_path) do
      {:ok, %{size: size}} -> size
      _ -> nil
    end
  end

  # Determine data source based on path
  defp determine_source(file_path) do
    path_parts = String.split(file_path, "/")
    cond do
      Enum.member?(path_parts, "aarslev") -> :aarslev
      Enum.member?(path_parts, "knudjepsen") -> :knudjepsen
      true -> :unknown
    end
  end
end 