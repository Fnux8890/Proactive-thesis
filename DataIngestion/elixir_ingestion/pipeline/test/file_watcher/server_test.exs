defmodule FileWatcher.ServerTest do
  use ExUnit.Case, async: false
  import Mox

  alias FileWatcher.Server

  # Set up mocks to verify they're called with correct arguments
  setup :verify_on_exit!

  setup do
    # Create a list of paths to watch for testing
    watch_paths = ["/app/data"]

    # Create a unique name for the StateStore in this test
    state_store_name =
      Module.concat(__MODULE__, "StateStore_#{System.unique_integer([:positive])}")

    # Allow this test process to use the mocks
    Mox.allow(MockStateStore, self())

    # Mock FileSystem watcher - using meck for file system operations
    :meck.new(FileSystem, [:passthrough])
    :meck.expect(FileSystem, :start_link, fn _ -> {:ok, self()} end)
    :meck.expect(FileSystem, :subscribe, fn _ -> :ok end)

    # Mock File.stat!/2
    :meck.new(File, [:passthrough])

    # Default expectation - empty state
    expect(MockStateStore, :load_state, fn -> {:ok, %{}} end)

    # Start the server with the mock state store
    {:ok, server_pid} =
      start_supervised!({
        FileWatcher.Server,
        watch_paths: watch_paths, name: :test_server, state_store: MockStateStore
      })

    # Clean up mocks when test exits
    on_exit(fn ->
      :meck.unload(FileSystem)
      :meck.unload(File)
    end)

    # Return the test context
    {:ok,
     %{
       server_pid: server_pid,
       watch_paths: watch_paths
     }}
  end

  describe "start_link/1" do
    test "starts the server with watch paths", %{watch_paths: watch_paths} do
      # Make sure we expect a load_state call from the new server
      expect(MockStateStore, :load_state, fn -> {:ok, %{}} end)

      # Start a new server instance for this specific test
      {:ok, pid} =
        Server.start_link(
          watch_paths: watch_paths,
          name: :test_start_link_server
        )

      # Verify server is running
      assert Process.alive?(pid)

      # Verify FileSystem was started with correct arguments
      assert :meck.called(FileSystem, :start_link, [%{dirs: watch_paths}])

      # Clean up
      GenServer.stop(pid)
    end
  end

  describe "initialization with StateStore" do
    test "loads state from StateStore if available" do
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

      # Start a new server with our mocked StateStore
      {:ok, pid} =
        Server.start_link(
          watch_paths: ["/app/data"],
          state_store: MockStateStore,
          name: :test_load_state_server
        )

      # Verify stored files are accessible via get_files API
      assert {:ok, files} = GenServer.call(pid, :get_files)
      assert files == test_files

      # Clean up
      GenServer.stop(pid)
    end

    test "initializes with empty files on StateStore error" do
      # Mock StateStore.load_state/0 to return error
      expect(MockStateStore, :load_state, fn -> {:error, "Failed to load state"} end)

      # Start a new server with our mocked StateStore
      {:ok, pid} =
        Server.start_link(
          watch_paths: ["/app/data"],
          state_store: MockStateStore,
          name: :test_empty_state_server
        )

      # Verify files are empty
      assert {:ok, files} = GenServer.call(pid, :get_files)
      assert files == %{}

      # Clean up
      GenServer.stop(pid)
    end
  end

  describe "file event handling" do
    test "processes file creation events", %{server_pid: pid} do
      # Subscribe to server notifications
      GenServer.call(pid, {:subscribe, self()})

      # Create file info
      file_path = "/app/data/new_file.txt"

      file_info = %{
        size: 100,
        type: :regular,
        mtime: {{2023, 1, 1}, {12, 0, 0}}
      }

      # Mock File.stat! to return file info
      :meck.expect(File, :stat!, fn ^file_path, [:time] -> file_info end)
      :meck.expect(File, :exists?, fn ^file_path -> true end)
      :meck.expect(File, :regular?, fn ^file_path -> true end)

      # Mock StateStore to expect save_state call
      expect(MockStateStore, :save_state, fn files ->
        assert Map.has_key?(files, file_path)
        :ok
      end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:created]}}

      # Send event to server
      send(pid, file_event)

      # Wait briefly for async processing
      :timer.sleep(100)

      # Verify via API that file was added
      {:ok, files} = GenServer.call(pid, :get_files)
      assert Map.has_key?(files, file_path)

      # Verify subscriber was notified about the new file
      assert_received {:file_update, %{path: ^file_path}}
    end

    test "processes file modification events", %{server_pid: pid} do
      # Set up existing file
      file_path = "/app/data/existing_file.txt"

      # Start with a file already known to the server
      initial_file_data = %{
        "name" => "existing_file.txt",
        "size" => 100,
        "type" => "text",
        "mtime" => "2023-01-01T12:00:00Z"
      }

      # Add the file to server's state
      GenServer.call(pid, {:set_test_files, %{file_path => initial_file_data}})

      # Subscribe to server notifications
      GenServer.call(pid, {:subscribe, self()})

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
      :meck.expect(File, :exists?, fn ^file_path -> true end)
      :meck.expect(File, :regular?, fn ^file_path -> true end)

      # Mock StateStore to expect save_state call
      expect(MockStateStore, :save_state, fn files ->
        file_data = Map.get(files, file_path)
        # Updated size
        assert file_data["size"] == 200
        :ok
      end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:modified]}}

      # Send event to server
      send(pid, file_event)

      # Wait briefly for async processing
      :timer.sleep(100)

      # Verify via API that file was updated
      {:ok, files} = GenServer.call(pid, :get_files)
      assert Map.has_key?(files, file_path)
      # Updated size
      assert files[file_path]["size"] == 200

      # Verify subscriber was notified
      assert_received {:file_update, %{path: ^file_path}}
    end

    test "processes file deletion events", %{server_pid: pid} do
      # Set up existing file
      file_path = "/app/data/to_be_deleted.txt"

      # Start with a file already known to the server
      initial_file_data = %{
        "name" => "to_be_deleted.txt",
        "size" => 100,
        "type" => "text",
        "mtime" => "2023-01-01T12:00:00Z"
      }

      # Add the file to server's state
      GenServer.call(pid, {:set_test_files, %{file_path => initial_file_data}})

      # Subscribe to server notifications
      GenServer.call(pid, {:subscribe, self()})

      # Mock File.stat! to raise error (file doesn't exist)
      :meck.expect(File, :stat!, fn ^file_path, [:time] ->
        raise File.Error, reason: :enoent, path: file_path, action: "stat"
      end)

      # Mock StateStore to expect save_state call
      expect(MockStateStore, :save_state, fn files ->
        # File should be removed
        refute Map.has_key?(files, file_path)
        :ok
      end)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:deleted]}}

      # Send event to server
      send(pid, file_event)

      # Wait briefly for async processing
      :timer.sleep(100)

      # Verify via API that file was removed
      {:ok, files} = GenServer.call(pid, :get_files)
      refute Map.has_key?(files, file_path)

      # Verify subscriber was notified
      assert_received {:file_removed, ^file_path}
    end

    test "ignores events for files outside watch paths", %{server_pid: pid} do
      # Create file path outside watch paths
      file_path = "/other/path/file.txt"

      # Get current state
      {:ok, initial_files} = GenServer.call(pid, :get_files)

      # Create file event message
      file_event = {:file_event, self(), {file_path, [:created]}}

      # Send event to server
      send(pid, file_event)

      # Wait briefly for async processing
      :timer.sleep(100)

      # Verify via API that state remains unchanged
      {:ok, files} = GenServer.call(pid, :get_files)
      assert files == initial_files
    end
  end

  describe "subscriber management" do
    test "subscribe and unsubscribe", %{server_pid: pid} do
      # Subscribe current process
      GenServer.call(pid, {:subscribe, self()})

      # Verify subscription
      subscribers = GenServer.call(pid, :get_subscribers)
      assert MapSet.member?(subscribers, self())

      # Unsubscribe current process
      GenServer.call(pid, {:unsubscribe, self()})

      # Verify unsubscription
      subscribers = GenServer.call(pid, :get_subscribers)
      refute MapSet.member?(subscribers, self())
    end

    test "handles subscriber process exits", %{server_pid: pid} do
      # Start a process that will subscribe and then exit
      subscriber_pid =
        spawn(fn ->
          # Subscribe to server
          GenServer.call(pid, {:subscribe, self()})
          # Send message back to test process
          send(self(), :subscribed)
          # Wait briefly to ensure the subscription is processed
          :timer.sleep(10)
        end)

      # Wait for subscription to complete
      assert_receive :subscribed, 100

      # Wait for subscriber process to exit
      :timer.sleep(100)

      # Verify subscriber was automatically removed after exit
      subscribers = GenServer.call(pid, :get_subscribers)
      refute MapSet.member?(subscribers, subscriber_pid)
    end
  end

  describe "state saving" do
    test "save_state saves current state to StateStore", %{server_pid: pid} do
      # Set up test files
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        }
      }

      # Update server state with test files
      GenServer.call(pid, {:set_test_files, test_files})

      # Mock StateStore.save_state/1 expectation
      expect(MockStateStore, :save_state, fn files ->
        assert files == test_files
        :ok
      end)

      # Call save_state directly through GenServer call
      assert GenServer.call(pid, :save_state) == :ok
    end

    test "handles save_state message", %{server_pid: pid} do
      # Set up test files
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        }
      }

      # Update server state with test files
      GenServer.call(pid, {:set_test_files, test_files})

      # Mock StateStore.save_state/1 expectation
      expect(MockStateStore, :save_state, fn files ->
        assert files == test_files
        :ok
      end)

      # Send save_state message
      send(pid, :save_state)

      # Wait for processing
      :timer.sleep(100)
    end

    test "save_state returns error when StateStore fails", %{server_pid: pid} do
      # Mock StateStore.save_state/1 to return error
      expect(MockStateStore, :save_state, fn _ ->
        {:error, "Save failed"}
      end)

      # Call save_state through GenServer call
      assert GenServer.call(pid, :save_state) == {:error, "Save failed"}
    end
  end

  describe "get_files/0" do
    test "returns current files", %{server_pid: pid} do
      # Set up test files
      test_files = %{
        "/app/data/file1.txt" => %{
          "name" => "file1.txt",
          "size" => 100,
          "type" => "text",
          "mtime" => "2023-01-01T12:00:00Z"
        }
      }

      # Update server state with test files
      GenServer.call(pid, {:set_test_files, test_files})

      # Call get_files via GenServer call
      {:ok, files} = GenServer.call(pid, :get_files)

      # Verify files match
      assert files == test_files
    end

    test "returns error when server is not running" do
      # Stop server if it's running
      if pid = Process.whereis(:test_server) do
        GenServer.stop(pid)
        # Wait for server to stop
        :timer.sleep(100)
      end

      # Call get_files on the module function (which checks if server is running)
      assert Server.get_files() == {:error, :server_not_running}
    end
  end

  describe "unexpected messages" do
    test "handles unexpected messages without crashing", %{server_pid: pid} do
      # Get current state
      {:ok, initial_files} = GenServer.call(pid, :get_files)

      # Send unexpected message
      send(pid, {:unexpected_message, "This should be ignored"})

      # Wait briefly
      :timer.sleep(100)

      # Verify via API that state remains unchanged
      {:ok, files} = GenServer.call(pid, :get_files)
      assert files == initial_files
    end
  end
end
