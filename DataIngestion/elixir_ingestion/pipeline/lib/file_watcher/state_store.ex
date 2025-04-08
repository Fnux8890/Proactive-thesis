defmodule FileWatcher.StateStore do
  @moduledoc """
  Stores and retrieves file watcher state in Redis.
  Provides atomic state storage operations with retry functionality.
  """
  @behaviour FileWatcher.StateStoreBehaviour
  use GenServer

  require Logger
  alias ConnectionHandler.Client, as: Redis
  alias Pipeline.Utils.Retry

  @files_prefix "file_watcher:files:"
  @files_set "file_watcher:files_set"
  @max_retries 3

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Loads the file watcher state from Redis.

  ## Returns
    * {:ok, state} - Map of file information by path
    * {:error, reason} - If loading failed
  """
  @impl FileWatcher.StateStoreBehaviour
  def load_state do
    GenServer.call(__MODULE__, :load_state)
  end

  @doc """
  Saves the file watcher state to Redis.
  Uses Redis transactions to ensure atomicity of operations.

  ## Parameters
    * files - Map of file information by path

  ## Returns
    * :ok - If the state was saved
    * {:error, reason} - If saving failed
  """
  @impl FileWatcher.StateStoreBehaviour
  def save_state(files) do
    GenServer.call(__MODULE__, {:save_state, files}, 30_000)
  end

  # Server callbacks

  @impl GenServer
  def init(_opts) do
    Logger.info("[FileWatcher.StateStore] Initializing...")
    {:ok, %{}}
  end

  @impl GenServer
  def handle_call(:load_state, _from, state) do
    Logger.debug("[FileWatcher.StateStore] handle_call(:load_state)")
    result = do_load_state()
    {:reply, result, state}
  end

  @impl GenServer
  def handle_call({:save_state, files}, _from, state) do
    Logger.debug("[FileWatcher.StateStore] handle_call(:save_state, #{map_size(files)} files)")
    result = do_save_state(files)
    {:reply, result, state}
  end

  # Private implementation

  defp do_load_state do
    Logger.debug("Loading file watcher state from Redis")

    try do
      # Get all paths from the set
      case Redis.smembers(@files_set) do
        {:ok, paths} when is_list(paths) and length(paths) > 0 ->
          # Construct keys for MGET
          keys = Enum.map(paths, fn path -> "#{@files_prefix}#{path}" end)

          case Redis.mget(keys) do
            {:ok, values} ->
              # Build state map from key/value pairs
              state =
                Enum.zip(paths, values)
                |> Enum.reduce(%{}, fn {path, json}, acc ->
                  case json do
                    nil ->
                      acc

                    _ ->
                      case Jason.decode(json) do
                        {:ok, file_info} ->
                          Map.put(acc, path, file_info)

                        {:error, reason} ->
                          Logger.warning(
                            "Failed to decode file info for #{path}: #{inspect(reason)}"
                          )

                          acc
                      end
                  end
                end)

              Logger.info("Loaded file watcher state: #{map_size(state)} files")
              {:ok, state}

            {:error, reason} ->
              Logger.error("Failed to load file values from Redis: #{inspect(reason)}")
              {:error, reason}
          end

        {:ok, []} ->
          Logger.info("No file watcher state found in Redis")
          {:ok, %{}}

        {:error, reason} ->
          Logger.error("Failed to load paths from Redis: #{inspect(reason)}")
          {:error, reason}
      end
    rescue
      e ->
        Logger.error("Exception loading file watcher state: #{inspect(e)}")
        {:error, e}
    end
  end

  defp do_save_state(files) do
    Logger.debug("Saving file watcher state to Redis: #{map_size(files)} files")

    try do
      # Get existing paths from Redis SET rather than using KEYS
      case Redis.smembers(@files_set) do
        {:ok, existing_paths} ->
          # Find paths that should be deleted (in Redis but not in state)
          current_paths = MapSet.new(Map.keys(files))
          existing_paths_set = MapSet.new(existing_paths)
          to_delete = MapSet.difference(existing_paths_set, current_paths)

          # Build Redis transaction pipeline
          cmds = [["MULTI"]]

          # Add commands to delete files that are no longer in state
          cmds =
            Enum.reduce(to_delete, cmds, fn path, acc ->
              Logger.debug("Removing file from Redis: #{path}")
              key = "#{@files_prefix}#{path}"
              acc ++ [["DEL", key], ["SREM", @files_set, path]]
            end)

          # Add commands to save/update file information
          cmds =
            Enum.reduce(files, cmds, fn {path, file_info}, acc ->
              key = "#{@files_prefix}#{path}"
              # JSON encode file info
              case Jason.encode(file_info) do
                {:ok, json} ->
                  acc ++ [["SET", key, json], ["SADD", @files_set, path]]

                {:error, reason} ->
                  Logger.error("Failed to encode file info for #{path}: #{inspect(reason)}")
                  acc
              end
            end)

          # Complete transaction with EXEC
          cmds = cmds ++ [["EXEC"]]

          # Execute pipeline with retries
          case Retry.retry_with_backoff(
                 fn -> Redis.pipeline(cmds) end,
                 "Redis transaction for save_state",
                 max_retries: @max_retries,
                 initial_delay: 100
               ) do
            {:ok, results} ->
              # Check if transaction was successful (last result is from EXEC)
              exec_result = List.last(results)

              if exec_result == nil do
                Logger.error("Redis transaction failed: EXEC returned nil")
                {:error, :transaction_failed}
              else
                Logger.info("Successfully saved file watcher state: #{map_size(files)} files")
                :ok
              end

            {:error, reason} ->
              Logger.error("Failed to execute Redis transaction: #{inspect(reason)}")
              {:error, reason}
          end

        {:error, reason} ->
          Logger.error("Failed to get existing file paths from Redis: #{inspect(reason)}")
          {:error, reason}
      end
    rescue
      e ->
        Logger.error("Exception saving file watcher state: #{inspect(e)}")
        {:error, e}
    end
  end
end
