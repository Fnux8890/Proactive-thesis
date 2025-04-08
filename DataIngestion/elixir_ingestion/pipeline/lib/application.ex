defmodule Pipeline.Application do
  @moduledoc """
  Main application for the data ingestion pipeline.
  """

  use Application
  require Logger

  def start(_type, _args) do
    # Only start the application's supervision tree if configured to do so
    # This allows tests to control startup manually
    start_supervision = Application.get_env(:pipeline, :start_supervision, true)

    if start_supervision do
      do_start()
    else
      Logger.info("Skipping application startup (start_supervision is false)")
      {:ok, self()}
    end
  end

  defp do_start do
    Logger.info("Starting Data Ingestion Pipeline Application")

    # Start the main application supervisor
    Pipeline.Supervisor.start_link([])
  end
end
