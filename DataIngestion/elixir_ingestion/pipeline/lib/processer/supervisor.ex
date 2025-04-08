defmodule Processor.Supervisor do
  @moduledoc """
  Dynamic Supervisor for managing FileProcessor workers.
  """
  use DynamicSupervisor
  require Logger

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    DynamicSupervisor.start_link(__MODULE__, [], name: name)
  end

  @impl true
  def init(_init_arg) do
    Logger.info("[Processor.Supervisor] Initializing...")
    Logger.info("Starting Processor Supervisor")
    # Restart workers temporarily if they crash, but don't restart indefinitely
    DynamicSupervisor.init(
      strategy: :one_for_one,
      max_restarts: 5,
      max_seconds: 60
    )
  end
end
