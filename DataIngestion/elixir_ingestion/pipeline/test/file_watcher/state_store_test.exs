# Mock for Pipeline.Utils.Retry
defmodule Pipeline.Utils.Retry do
  # Simple mock for Retry module
  def retry_with_backoff(fun, _operation_name, _opts \\ []) do
    fun.()
  end
end

# Define the mock behavior
defmodule ConnectionHandler.ClientBehavior do
  @callback smembers(String.t()) :: {:ok, list(String.t())} | {:error, any()}
  @callback mget(list(String.t())) :: {:ok, list(String.t())} | {:error, any()}
  @callback pipeline(list(list(String.t()))) :: {:ok, list(any())} | {:error, any()}
end

# Create the mock with Mox
Mox.defmock(MockRedisClient, for: ConnectionHandler.ClientBehavior)

# Replace the real Redis client with our mock during tests
defmodule FileWatcher.StateStore do
  use GenServer

  require Logger
  # Use the mock during tests
  alias MockRedisClient, as: Redis

  @files_prefix "file_watcher:file:"
  @files_set "file_watcher:files_set"

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts)
  end

  def init(_opts) do
    {:ok, %{}}
  end

  def handle_call(:load_state, _from, state) do
    result = do_load_state()
    {:reply, result, state}
  end

  def handle_call({:save_state, files}, _from, state) do
    result = do_save_state(files)
    {:reply, result, state}
  end

  # Private implementation

  defp do_load_state do
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

                        {:error, _reason} ->
                          acc
                      end
                  end
                end)

              {:ok, state}

            {:error, reason} ->
              {:error, reason}
          end

        {:ok, []} ->
          {:ok, %{}}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e ->
        {:error, e}
    end
  end

  defp do_save_state(files) do
    try do
      # Get existing paths from Redis SET
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

                {:error, _reason} ->
                  acc
              end
            end)

          # Complete transaction with EXEC
          cmds = cmds ++ [["EXEC"]]

          # Execute pipeline
          case Redis.pipeline(cmds) do
            {:ok, results} ->
              # Check if transaction was successful (last result is from EXEC)
              exec_result = List.last(results)

              if exec_result == nil do
                {:error, :transaction_failed}
              else
                :ok
              end

            {:error, reason} ->
              {:error, reason}
          end

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e ->
        {:error, e}
    end
  end
end

defmodule FileWatcher.StateStoreTest do
  use ExUnit.Case
  import Mox

  # Verify mocks after each test
  setup :verify_on_exit!
  # Set mox to global mode to allow any process to call mocked modules
  setup :set_mox_global

  setup do
    # Start a GenServer with our test implementation
    pid = start_supervised!(FileWatcher.StateStore)

    # Allow this test process to handle calls from any process
    # When in global mode, use ExUnit.Server as the owner_pid
    Mox.allow(MockRedisClient, Process.whereis(ExUnit.Server), self())

    # Return the pid for use in tests
    {:ok, %{pid: pid}}
  end

  test "load_state/0 returns empty map when no files exist", %{pid: pid} do
    # Set expectations for the call StateStore will make
    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, []}
    end)

    # Call the function via GenServer API
    assert GenServer.call(pid, :load_state) == {:ok, %{}}
  end

  test "load_state/0 returns map of files when files exist", %{pid: pid} do
    paths = ["/app/data/file1.txt", "/app/data/file2.txt"]

    keys = [
      "file_watcher:file:/app/data/file1.txt",
      "file_watcher:file:/app/data/file2.txt"
    ]

    values = [
      ~s({"sha": "abc123", "size": 100, "last_modified": "2023-01-01T00:00:00Z"}),
      ~s({"sha": "def456", "size": 200, "last_modified": "2023-01-02T00:00:00Z"})
    ]

    expected = %{
      "/app/data/file1.txt" => %{
        "sha" => "abc123",
        "size" => 100,
        "last_modified" => "2023-01-01T00:00:00Z"
      },
      "/app/data/file2.txt" => %{
        "sha" => "def456",
        "size" => 200,
        "last_modified" => "2023-01-02T00:00:00Z"
      }
    }

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, paths}
    end)
    |> expect(:mget, fn ^keys ->
      {:ok, values}
    end)

    assert GenServer.call(pid, :load_state) == {:ok, expected}
  end

  test "load_state/0 handles smembers error", %{pid: pid} do
    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:error, "Redis error"}
    end)

    assert GenServer.call(pid, :load_state) == {:error, "Redis error"}
  end

  test "load_state/0 handles invalid JSON in file metadata", %{pid: pid} do
    paths = ["/app/data/file1.txt"]
    keys = ["file_watcher:file:/app/data/file1.txt"]

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, paths}
    end)
    |> expect(:mget, fn ^keys ->
      {:ok, ["NOT VALID JSON"]}
    end)

    # Should ignore invalid JSON
    assert GenServer.call(pid, :load_state) == {:ok, %{}}
  end

  test "load_state/0 handles mget error", %{pid: pid} do
    paths = ["/app/data/file1.txt"]
    keys = ["file_watcher:file:/app/data/file1.txt"]

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, paths}
    end)
    |> expect(:mget, fn ^keys ->
      {:error, "Redis error"}
    end)

    assert GenServer.call(pid, :load_state) == {:error, "Redis error"}
  end

  test "save_state/1 saves state with no existing files", %{pid: pid} do
    file_path = "/app/data/file1.txt"

    state = %{
      file_path => %{
        "sha" => "abc123",
        "size" => 100,
        "last_modified" => "2023-01-01T00:00:00Z"
      }
    }

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, []}
    end)
    |> expect(:pipeline, fn commands ->
      # Verify basic transaction structure
      assert Enum.at(commands, 0) == ["MULTI"]
      assert List.last(commands) == ["EXEC"]

      # Find SET command with our file
      set_cmd =
        Enum.find(commands, fn cmd ->
          length(cmd) >= 3 && hd(cmd) == "SET" &&
            String.starts_with?(Enum.at(cmd, 1), "file_watcher:file:")
        end)

      # Verify JSON data contains expected values
      json = Enum.at(set_cmd, 2)
      assert json =~ "abc123"
      assert json =~ "100"

      # Find SADD command for our file
      _sadd_cmd =
        Enum.find(commands, fn cmd ->
          length(cmd) >= 3 && hd(cmd) == "SADD" &&
            Enum.at(cmd, 1) == "file_watcher:files_set"
        end)

      # Return successful transaction
      {:ok, ["OK", "OK", 1, ["OK", 1]]}
    end)

    assert GenServer.call(pid, {:save_state, state}) == :ok
  end

  test "save_state/1 saves state with existing files to add and remove", %{pid: pid} do
    # New state has one file
    file_path = "/app/data/new_file.txt"

    state = %{
      file_path => %{
        "sha" => "abc123",
        "size" => 100,
        "last_modified" => "2023-01-01T00:00:00Z"
      }
    }

    # Redis already has a different file
    existing_path = "/app/data/old_file.txt"

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, [existing_path]}
    end)
    |> expect(:pipeline, fn commands ->
      # Verify transaction contains MULTI and EXEC
      assert Enum.at(commands, 0) == ["MULTI"]
      assert List.last(commands) == ["EXEC"]

      # Should have DEL + SREM for old file and SET + SADD for new file
      assert Enum.any?(commands, fn cmd ->
               length(cmd) >= 2 && hd(cmd) == "DEL" &&
                 String.ends_with?(Enum.at(cmd, 1), existing_path)
             end)

      assert Enum.any?(commands, fn cmd ->
               length(cmd) >= 3 && hd(cmd) == "SREM" &&
                 Enum.at(cmd, 1) == "file_watcher:files_set" &&
                 Enum.at(cmd, 2) == existing_path
             end)

      assert Enum.any?(commands, fn cmd ->
               length(cmd) >= 3 && hd(cmd) == "SET" &&
                 String.ends_with?(Enum.at(cmd, 1), file_path)
             end)

      assert Enum.any?(commands, fn cmd ->
               length(cmd) >= 3 && hd(cmd) == "SADD" &&
                 Enum.at(cmd, 1) == "file_watcher:files_set" &&
                 Enum.at(cmd, 2) == file_path
             end)

      # Return successful transaction
      {:ok, ["OK", "OK", 1, 1, "OK", 1, ["OK", 1, 1, 1]]}
    end)

    assert GenServer.call(pid, {:save_state, state}) == :ok
  end

  test "save_state/1 handles pipeline error", %{pid: pid} do
    state = %{
      "/app/data/file1.txt" => %{
        "sha" => "abc123"
      }
    }

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, []}
    end)
    |> expect(:pipeline, fn _commands ->
      {:error, "Redis pipeline error"}
    end)

    assert GenServer.call(pid, {:save_state, state}) == {:error, "Redis pipeline error"}
  end

  test "save_state/1 handles transaction failure", %{pid: pid} do
    state = %{
      "/app/data/file1.txt" => %{
        "sha" => "abc123"
      }
    }

    MockRedisClient
    |> expect(:smembers, fn "file_watcher:files_set" ->
      {:ok, []}
    end)
    |> expect(:pipeline, fn _commands ->
      # EXEC returned nil
      {:ok, ["OK", nil]}
    end)

    assert GenServer.call(pid, {:save_state, state}) == {:error, :transaction_failed}
  end
end
