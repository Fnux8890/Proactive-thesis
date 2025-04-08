defmodule SchemaInference.Server do
  @moduledoc """
  GenServer responsible for consuming parsed data batches,
  performing schema inference (placeholder), and potentially passing data onwards.
  """
  use GenServer
  require Logger

  # --- Configuration ---
  @redis_queue_key "parsed_data_queue" # Key for the list in Redis
  @redis_client ConnectionHandler.Client # Assuming this is the configured Redis client
  @check_interval_ms 1000 # How often to check the queue if idle (milliseconds)
  @redis_block_timeout_sec 1 # How long BRPOP should block (seconds)

  # --- Client API ---
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, [], name: name)
  end

  # --- Server Callbacks ---
  @impl true
  def init(_opts) do
    Logger.info("[SchemaInference.Server] Initializing...")
    # Start checking the queue shortly after starting
    schedule_queue_check(50) # Start quickly
    {:ok, %{}}
  end

  @impl true
  def handle_info(:check_queue, state) do
    # Use BRPOP via the generic command function
    case @redis_client.command(["BRPOP", @redis_queue_key, @redis_block_timeout_sec]) do
      {:ok, [_queue_key, json_data]} when not is_nil(json_data) -> # BRPOP returns [key, value] on success
        # Data received from queue
        Logger.debug("[SchemaInference.Server] Received data batch from Redis.")
        try do
          # Deserialize the JSON data
          parsed_data = Jason.decode!(json_data) # Use decode! for simplicity, handle errors if needed

          record_count = if is_list(parsed_data), do: length(parsed_data), else: 1
          Logger.info("[SchemaInference.Server] Successfully deserialized batch of #{record_count} records.")

          # --- Placeholder for actual schema inference logic ---
          # infer_schema(parsed_data)
          # ------------------------------------------------------

          # TODO: Send inferred schema or data + schema to the next stage (e.g., DataProfiler)

          # Immediately check again in case more data is waiting
          send(self(), :check_queue)
          {:noreply, state}
        rescue
          e in Jason.DecodeError ->
            Logger.error("[SchemaInference.Server] Failed to decode JSON data: #{inspect(e)}. Data: #{inspect(json_data)}")
            schedule_queue_check(@check_interval_ms) # Schedule next check after error
            {:noreply, state}
          e ->
            Logger.error("[SchemaInference.Server] Unexpected error processing batch: #{inspect(e)}. Data: #{inspect(json_data)}")
            schedule_queue_check(@check_interval_ms) # Schedule next check after error
            {:noreply, state}
        end

      {:ok, nil} -> # BRPOP returns nil value in the list on timeout: {:ok, [key, nil]} or just {:ok, nil}
        # Queue was empty during the blocking timeout
        Logger.debug("[SchemaInference.Server] Parsed data queue empty or timeout, scheduling next check.")
        schedule_queue_check(@check_interval_ms)
        {:noreply, state}

      {:error, reason} ->
        # Error communicating with Redis
        Logger.error("[SchemaInference.Server] Redis error checking queue: #{inspect(reason)}")
        schedule_queue_check(@check_interval_ms * 2) # Wait longer after Redis error
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warning("[SchemaInference.Server] Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  defp schedule_queue_check(interval_ms) do
    Process.send_after(self(), :check_queue, interval_ms)
  end
end
