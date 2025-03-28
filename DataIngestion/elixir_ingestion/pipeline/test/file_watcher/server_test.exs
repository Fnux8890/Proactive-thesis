defmodule FileWatcher.ServerTest do
  use ExUnit.Case, async: false
  import Mox

  alias FileWatcher.Server

  # Set up mocks to verify they're called with correct arguments
  setup :verify_on_exit!

  setup do
    # Create a list of paths to watch for testing
    watch_paths = ["/app/data"]

    # Set up state for the tests
    state = %{
      files: %{},
      watch_paths: watch_paths,
      subscribers: MapSet.new(),
      file_system_pid: nil
    }

    # Mock the file_system watcher
    :meck.new(FileSystem, [:passthrough])
    :meck.expect(FileSystem, :start_link, fn _ -> {:ok, self()} end)
    :meck.expect(FileSystem, :subscribe, fn _ -> :ok end)

    # Mock File.stat!/2
    :meck.new(File, [:passthrough])

    on_exit(fn ->
      :meck.unload(FileSystem)
      :meck.unload(File)
    end)

    {:ok, %{state: state, watch_paths: watch_paths}}
  end

  describe "start_link/1" do
    test "starts the server with watch paths", %{watch_paths: watch_paths} do
      # Start the server
      {:ok, pid} = Server.start_link(watch_paths: watch_paths)

      # Verify server is running
      assert Process.alive?(pid)

      # Verify FileSystem was started with correct arguments
      assert :meck.called(FileSystem, :start_link, [%{dirs: watch_paths}])

      # Verify FileSystem.subscribe was called
      assert :meck.called(FileSystem, :subscribe, [:_])

      # Clean up
      GenServer.stop(pid)
    end
  end

  describe "init/1" do
    test "initializes state correctly", %{watch_paths: watch_paths} do
      # Directly call init with options
      {:ok, state} = Server.init(watch_paths: watch_paths)

      # Verify state
      assert state.watch_paths == watch_paths
      assert state.files == %{}
      assert state.subscribers == MapSet.new()
      assert state.file_system_pid != nil

      # Verify FileSystem was started
      assert :meck.called(FileSystem, :start_link, [%{dirs: watch_paths}])
    end

    test "loads state from StateStore if available", %{watch_paths: watch_paths} do
      # Define test files
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        }
      }

      # Mock StateStore.load_state/0 to return files
      expect(MockStateStore, :load_state, fn -> {:ok, test_files} end)

      # Call init
      {:ok, state} = Server.init(watch_paths: watch_paths, state_store: MockStateStore)

      # Verify state includes loaded files
      assert state.files == test_files
    end

    test "initializes with empty files on StateStore error", %{watch_paths: watch_paths} do
      # Mock StateStore.load_state/0 to return error
      expect(MockStateStore, :load_state, fn -> {:error, "Failed to load state"} end)

      # Call init
      {:ok, state} = Server.init(watch_paths: watch_paths, state_store: MockStateStore)

      # Verify state has empty files
      assert state.files == %{}
    end
  end

  describe "handle_info/2 for file events" do
    test "processes file creation events", %{state: state} do
      # Create file info
      file_path = "/app/data/new_file.txt"

      file_info = %{
        size: 100,
        type: :regular,
        mtime: {{2023, 1, 1}, {12, 0, 0}}
      }

      # Mock File.stat! to return file info
      :meck.expect(File, :stat!, fn ^file_path, [:time] -> file_info end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:created]}}

      # Call handle_info with the file event
      {:noreply, new_state} = Server.handle_info(file_event, state)

      # Verify file was added to state
      assert Map.has_key?(new_state.files, file_path)
      file_data = new_state.files[file_path]
      assert file_data["size"] == 100
    end

    test "processes file modification events", %{state: state} do
      # Set up existing file in state
      file_path = "/app/data/existing_file.txt"

      initial_state = %{
        state
        | files: %{
            file_path => %{
              "name" => "existing_file.txt",
              "size" => 100,
              "type" => "text",
              "mtime" => "2023-01-01T12:00:00Z"
            }
          }
      }

      # Create updated file info
      file_info = %{
        # Size changed
        size: 200,
        type: :regular,
        # Time changed
        mtime: {{2023, 1, 1}, {13, 0, 0}}
      }

      # Mock File.stat! to return updated file info
      :meck.expect(File, :stat!, fn ^file_path, [:time] -> file_info end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:modified]}}

      # Call handle_info with the file event
      {:noreply, new_state} = Server.handle_info(file_event, initial_state)

      # Verify file was updated in state
      assert Map.has_key?(new_state.files, file_path)
      file_data = new_state.files[file_path]
      # Updated size
      assert file_data["size"] == 200
    end

    test "processes file deletion events", %{state: state} do
      # Set up existing file in state
      file_path = "/app/data/to_be_deleted.txt"

      initial_state = %{
        state
        | files: %{
            file_path => %{
              "name" => "to_be_deleted.txt",
              "size" => 100,
              "type" => "text",
              "mtime" => "2023-01-01T12:00:00Z"
            }
          }
      }

      # Mock File.stat! to raise error (file doesn't exist)
      :meck.expect(File, :stat!, fn ^file_path, [:time] ->
        raise File.Error, reason: :enoent, path: file_path, action: "stat"
      end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:deleted]}}

      # Call handle_info with the file event
      {:noreply, new_state} = Server.handle_info(file_event, initial_state)

      # Verify file was removed from state
      refute Map.has_key?(new_state.files, file_path)
    end

    test "ignores events for files outside watch paths", %{state: state} do
      # Create file path outside watch paths
      file_path = "/other/path/file.txt"

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:created]}}

      # Call handle_info with the file event
      {:noreply, new_state} = Server.handle_info(file_event, state)

      # Verify state remains unchanged
      assert new_state == state
    end

    test "notifies subscribers about file updates", %{state: state} do
      # Add a subscriber (this test process)
      subscriber_pid = self()
      initial_state = %{state | subscribers: MapSet.new([subscriber_pid])}

      # Create file info
      file_path = "/app/data/new_file.txt"

      file_info = %{
        size: 100,
        type: :regular,
        mtime: {{2023, 1, 1}, {12, 0, 0}}
      }

      # Mock File.stat! to return file info
      :meck.expect(File, :stat!, fn ^file_path, [:time] -> file_info end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:created]}}

      # Call handle_info with the file event
      {:noreply, _new_state} = Server.handle_info(file_event, initial_state)

      # Verify subscriber was notified
      assert_received {:file_update, %{path: ^file_path}}
    end

    test "notifies subscribers about file deletions", %{state: state} do
      # Add a subscriber (this test process)
      subscriber_pid = self()

      # Set up existing file in state
      file_path = "/app/data/to_be_deleted.txt"

      initial_state = %{
        state
        | subscribers: MapSet.new([subscriber_pid]),
          files: %{
            file_path => %{
              "name" => "to_be_deleted.txt",
              "size" => 100,
              "type" => "text",
              "mtime" => "2023-01-01T12:00:00Z"
            }
          }
      }

      # Mock File.stat! to raise error (file doesn't exist)
      :meck.expect(File, :stat!, fn ^file_path, [:time] ->
        raise File.Error, reason: :enoent, path: file_path, action: "stat"
      end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:deleted]}}

      # Call handle_info with the file event
      {:noreply, _new_state} = Server.handle_info(file_event, initial_state)

      # Verify subscriber was notified
      assert_received {:file_removed, ^file_path}
    end
  end

  describe "get_files/0" do
    test "returns current files" do
      # Set up test files
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        },
        "/app/data/file2.txt" => %{
          "name" => "file2.txt",
          "size" => 200,
          "type" => "text",
          "mtime" => "2023-01-01T13:00:00Z"
        }
      }

      # Start server with test files in state
      {:ok, pid} = Server.start_link(watch_paths: ["/app/data"])
      :sys.replace_state(pid, fn state -> %{state | files: test_files} end)

      # Call get_files
      {:ok, files} = Server.get_files()

      # Verify files match
      assert files == test_files

      # Clean up
      GenServer.stop(pid)
    end

    test "returns error when server is not running" do
      # Stop server if it's running
      if pid = Process.whereis(FileWatcher.Server) do
        GenServer.stop(pid)
        # Wait for server to stop
        :timer.sleep(10)
      end

      # Call get_files
      assert Server.get_files() == {:error, :server_not_running}
    end
  end

  describe "subscribe/1 and unsubscribe/1" do
    test "adds and removes subscribers" do
      # Start server
      {:ok, pid} = Server.start_link(watch_paths: ["/app/data"])

      # Subscribe current process
      :ok = Server.subscribe(self())

      # Check that state includes the subscriber
      state = :sys.get_state(pid)
      assert MapSet.member?(state.subscribers, self())

      # Unsubscribe current process
      :ok = Server.unsubscribe(self())

      # Check that state no longer includes the subscriber
      state = :sys.get_state(pid)
      refute MapSet.member?(state.subscribers, self())

      # Clean up
      GenServer.stop(pid)
    end

    test "handles subscriber process exits" do
      # Start server
      {:ok, server_pid} = Server.start_link(watch_paths: ["/app/data"])

      # Start a process that will subscribe and then exit
      subscriber_pid =
        spawn(fn ->
          # Subscribe to server
          Server.subscribe(self())
          # Send message back to test process
          send(self(), :subscribed)
          # Wait briefly to ensure the subscription is processed
          :timer.sleep(10)
        end)

      # Wait for subscription to complete
      assert_receive :subscribed, 100

      # Wait for subscriber process to exit
      :timer.sleep(20)

      # Verify subscriber was automatically removed after exit
      state = :sys.get_state(server_pid)
      refute MapSet.member?(state.subscribers, subscriber_pid)

      # Clean up
      GenServer.stop(server_pid)
    end
  end

  describe "save_state/0" do
    test "saves current state to StateStore" do
      # Set up test files
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        }
      }

      # Start server with test files and mock state store
      {:ok, pid} =
        Server.start_link(
          watch_paths: ["/app/data"],
          state_store: MockStateStore
        )

      # Update server state with test files
      :sys.replace_state(pid, fn state -> %{state | files: test_files} end)

      # Mock StateStore.save_state/1 expectation
      expect(MockStateStore, :save_state, fn files ->
        assert files == test_files
        :ok
      end)

      # Call save_state
      assert Server.save_state() == :ok

      # Clean up
      GenServer.stop(pid)
    end

    test "returns error when StateStore fails" do
      # Start server with mock state store
      {:ok, pid} =
        Server.start_link(
          watch_paths: ["/app/data"],
          state_store: MockStateStore
        )

      # Mock StateStore.save_state/1 to return error
      expect(MockStateStore, :save_state, fn _ ->
        {:error, "Save failed"}
      end)

      # Call save_state
      assert Server.save_state() == {:error, "Save failed"}

      # Clean up
      GenServer.stop(pid)
    end
  end

  describe "handle_info/2 for non-file events" do
    test "ignores unexpected messages", %{state: state} do
      # Send unexpected message
      unexpected_message = {:unexpected, :message}

      # Call handle_info with unexpected message
      {:noreply, new_state} = Server.handle_info(unexpected_message, state)

      # Verify state unchanged
      assert new_state == state
    end

    test "processes :save_state message", %{state: state} do
      # Add mock state store to state
      state = Map.put(state, :state_store, MockStateStore)

      # Add test files to state
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        }
      }

      state = %{state | files: test_files}

      # Mock StateStore.save_state/1 expectation
      expect(MockStateStore, :save_state, fn files ->
        assert files == test_files
        :ok
      end)

      # Create save_state message
      message = :save_state

      # Call handle_info with save_state message
      {:noreply, new_state} = Server.handle_info(message, state)

      # Verify state unchanged (save doesn't modify state)
      assert new_state == state
    end
  end
end
