defmodule Producer.ProducerBehaviour do
  @moduledoc """
  Behaviour for the FileQueueProducer interface used by callers.
  """

  @doc """
  Starts a new producer process.

  ## Parameters
    * opts - Keyword list or map of options to initialize the producer

  ## Returns
    * `{:ok, pid}` - If the process started successfully
    * `{:error, reason}` - If the process failed to start
  """
  @callback start_link(opts :: Keyword.t() | map()) :: GenServer.on_start()

  @doc """
  Enqueues a file for processing.

  ## Parameters
    * producer - The producer process or name
    * file_path - The path to the file to be processed
    * metadata - Additional metadata about the file

  ## Returns
    * `{:ok, file_id}` - If the file was enqueued successfully
    * `{:error, reason}` - If the file could not be enqueued
  """
  @callback enqueue_file(
              producer :: GenServer.server(),
              file_path :: String.t(),
              metadata :: map()
            ) ::
              {:ok, String.t()} | {:error, atom()}

  @doc """
  Gets the current status of the producer's queues.

  ## Parameters
    * producer - The producer process or name

  ## Returns
    * A map containing queue statistics
  """
  @callback queue_status(producer :: GenServer.server()) :: map()
end
