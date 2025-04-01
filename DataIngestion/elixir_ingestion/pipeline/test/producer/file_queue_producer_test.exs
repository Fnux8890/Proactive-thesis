defmodule Producer.FileQueueProducerTest do
  use ExUnit.Case, async: false
  import Mox

  alias Producer.FileQueueProducer

  # Make sure mocks are verified after each test
  setup :verify_on_exit!

  setup do
    # Set attributes for testing
    attrs = %{
      name: :file_queue_producer_test,
      buffer_size: 10,
      min_demand: 1,
      max_demand: 5,
      dispatch_method: :push,
      file_timeout: 10_000
    }

    # Mock UUID generation to return predictable values
    :meck.new(UUID, [:passthrough])
    :meck.expect(UUID, :uuid4, fn -> "mock-uuid" end)

    # Mock Process.send_after for timeout check scheduling
    :meck.new(Process, [:passthrough])

    :meck.expect(Process, :send_after, fn pid, msg, timeout ->
      make_ref()
    end)

    # Set up state for tests that need it
    state = %{
      demand: 0,
      events: [],
      buffer_size: attrs.buffer_size,
      name: attrs.name,
      dispatch_method: attrs.dispatch_method,
      in_progress_files: %{},
      file_timeout: attrs.file_timeout
    }

    on_exit(fn ->
      :meck.unload(UUID)
      :meck.unload(Process)
    end)

    {:ok, %{attrs: attrs, state: state}}
  end

  describe "start_link/1" do
    test "starts the GenStage producer", %{attrs: attrs} do
      # Start the producer
      {:ok, pid} = FileQueueProducer.start_link(attrs)

      # Verify it's alive
      assert Process.alive?(pid)

      # Check that it has the right name
      assert Process.whereis(attrs.name) == pid

      # Clean up
      GenStage.stop(pid)
    end
  end

  describe "init/1" do
    test "initializes state correctly", %{attrs: attrs} do
      # Call init directly
      {:producer, state, _} = FileQueueProducer.init(attrs)

      # Verify state
      assert state.demand == 0
      assert state.events == []
      assert state.buffer_size == attrs.buffer_size
      assert state.name == attrs.name
      assert state.dispatch_method == attrs.dispatch_method
      assert state.in_progress_files == %{}
      assert state.file_timeout == attrs.file_timeout

      # Verify timeout check was scheduled
      assert :meck.called(Process, :send_after, [:_, :check_timeouts, :_])
    end
  end

  describe "handle_demand/2" do
    test "stores demand when no events available", %{state: state} do
      # Call handle_demand with demand of 5
      {:noreply, [], new_state} = FileQueueProducer.handle_demand(5, state)

      # Verify demand was stored
      assert new_state.demand == 5
    end

    test "dispatches events up to demand", %{state: state} do
      # Add events to state
      state = %{
        state
        | events: [
            %{id: "file1", path: "/path/to/file1", metadata: %{size: 100}},
            %{id: "file2", path: "/path/to/file2", metadata: %{size: 200}},
            %{id: "file3", path: "/path/to/file3", metadata: %{size: 300}}
          ]
      }

      # Call handle_demand with demand of 2
      {:noreply, dispatched, new_state} = FileQueueProducer.handle_demand(2, state)

      # Verify correct events were dispatched
      assert length(dispatched) == 2
      assert Enum.at(dispatched, 0).id == "file1"
      assert Enum.at(dispatched, 1).id == "file2"

      # Verify remaining event is in state
      assert length(new_state.events) == 1
      assert Enum.at(new_state.events, 0).id == "file3"

      # Verify demand is 0 (fully satisfied)
      assert new_state.demand == 0

      # Verify in_progress_files was updated
      assert Map.has_key?(new_state.in_progress_files, "file1")
      assert Map.has_key?(new_state.in_progress_files, "file2")
    end

    test "handles demand greater than available events", %{state: state} do
      # Add events to state
      state = %{
        state
        | events: [
            %{id: "file1", path: "/path/to/file1", metadata: %{size: 100}},
            %{id: "file2", path: "/path/to/file2", metadata: %{size: 200}}
          ]
      }

      # Call handle_demand with demand of 5
      {:noreply, dispatched, new_state} = FileQueueProducer.handle_demand(5, state)

      # Verify all events were dispatched
      assert length(dispatched) == 2

      # Verify events list is empty
      assert new_state.events == []

      # Verify remaining demand
      assert new_state.demand == 3

      # Verify in_progress_files was updated
      assert Map.has_key?(new_state.in_progress_files, "file1")
      assert Map.has_key?(new_state.in_progress_files, "file2")
    end
  end

  describe "handle_call/3 :enqueue_file" do
    test "enqueues file with generated UUID", %{state: state} do
      # Set up test data
      file_path = "/path/to/file.txt"
      metadata = %{size: 100, type: "text"}

      # Call handle_call with enqueue_file request
      result = FileQueueProducer.handle_call({:enqueue_file, file_path, metadata}, self(), state)

      # Verify result contains the UUID
      assert {:reply, {:ok, "mock-uuid"}, _, new_state} = result

      # Verify file was added to events
      assert length(new_state.events) == 1
      event = Enum.at(new_state.events, 0)
      assert event.id == "mock-uuid"
      assert event.path == file_path
      assert event.metadata.size == 100
      assert event.metadata.type == "text"

      # Verify UUID generation was called
      assert :meck.called(UUID, :uuid4, [])
    end

    test "dispatches immediately when demand exists", %{state: state} do
      # Set up state with demand
      state = %{state | demand: 2}

      # Set up test data
      file_path = "/path/to/file.txt"
      metadata = %{size: 100, type: "text"}

      # Call handle_call with enqueue_file request
      result = FileQueueProducer.handle_call({:enqueue_file, file_path, metadata}, self(), state)

      # Verify result contains the UUID
      assert {:reply, {:ok, "mock-uuid"}, dispatched, new_state} = result

      # Verify event was dispatched immediately
      assert length(dispatched) == 1
      event = Enum.at(dispatched, 0)
      assert event.id == "mock-uuid"
      assert event.path == file_path

      # Verify event was not stored in events list
      assert new_state.events == []

      # Verify demand was decreased
      assert new_state.demand == 1

      # Verify in_progress_files was updated
      assert Map.has_key?(new_state.in_progress_files, "mock-uuid")
    end

    test "respects buffer size limit", %{state: state} do
      # Fill events to buffer limit
      events =
        Enum.map(1..10, fn i ->
          %{id: "file#{i}", path: "/path/to/file#{i}", metadata: %{size: i * 100}}
        end)

      state = %{state | events: events}

      # Set up test data for new event
      file_path = "/path/to/overflow.txt"
      metadata = %{size: 1000, type: "text"}

      # Call handle_call with enqueue_file request
      result = FileQueueProducer.handle_call({:enqueue_file, file_path, metadata}, self(), state)

      # Verify error response
      assert {:reply, {:error, :queue_full}, _, _} = result
    end
  end

  describe "handle_call/3 :file_complete" do
    test "marks file as complete", %{state: state} do
      # Set up in_progress_files with a tracked file
      file_id = "file-123"

      state = %{
        state
        | in_progress_files: %{
            file_id => %{path: "/path/to/file.txt", enqueued_at: System.system_time(:millisecond)}
          }
      }

      # Call handle_call with file_complete request
      {:reply, :ok, [], new_state} =
        FileQueueProducer.handle_call({:file_complete, file_id}, self(), state)

      # Verify file was removed from in_progress_files
      refute Map.has_key?(new_state.in_progress_files, file_id)
    end

    test "handles unknown file ID", %{state: state} do
      # Call handle_call with unknown file ID
      {:reply, :ok, [], new_state} =
        FileQueueProducer.handle_call({:file_complete, "unknown-id"}, self(), state)

      # Verify state unchanged
      assert new_state.in_progress_files == state.in_progress_files
    end
  end

  describe "handle_call/3 :queue_state" do
    test "returns queue state info", %{state: state} do
      # Set up test state
      events = [
        %{id: "file1", path: "/path/to/file1", metadata: %{size: 100}},
        %{id: "file2", path: "/path/to/file2", metadata: %{size: 200}}
      ]

      in_progress = %{
        "file3" => %{path: "/path/to/file3", enqueued_at: System.system_time(:millisecond)},
        "file4" => %{path: "/path/to/file4", enqueued_at: System.system_time(:millisecond)}
      }

      state = %{state | events: events, in_progress_files: in_progress, demand: 3}

      # Call handle_call with queue_state request
      {:reply, queue_state, [], ^state} =
        FileQueueProducer.handle_call(:queue_state, self(), state)

      # Verify queue_state info
      assert queue_state.queued_count == 2
      assert queue_state.in_progress_count == 2
      assert queue_state.demand == 3
    end
  end

  describe "handle_info/2 :check_timeouts" do
    test "removes timed out files and reschedules check", %{state: state} do
      # Current time for testing
      now = System.system_time(:millisecond)

      # Set up in_progress_files with recent and old files
      in_progress = %{
        "recent" => %{
          path: "/path/to/recent.txt",
          # 5 seconds ago, under timeout
          enqueued_at: now - 5_000
        },
        "old" => %{
          path: "/path/to/old.txt",
          # 15 seconds ago, over timeout
          enqueued_at: now - 15_000
        }
      }

      state = %{state | in_progress_files: in_progress, file_timeout: 10_000}

      # Call handle_info with check_timeouts message
      {:noreply, [], new_state} = FileQueueProducer.handle_info(:check_timeouts, state)

      # Verify old file was removed but recent file remains
      assert Map.has_key?(new_state.in_progress_files, "recent")
      refute Map.has_key?(new_state.in_progress_files, "old")

      # Verify next check was scheduled
      assert :meck.called(Process, :send_after, [:_, :check_timeouts, :_])
    end

    test "re-enqueues timed out files when demand exists", %{state: state} do
      # Current time for testing
      now = System.system_time(:millisecond)

      # Set up in_progress_files with a timed out file
      file_id = "timed-out"
      file_path = "/path/to/timeout.txt"
      metadata = %{size: 100, type: "text"}

      in_progress = %{
        file_id => %{
          path: file_path,
          metadata: metadata,
          # 15 seconds ago, over timeout
          enqueued_at: now - 15_000
        }
      }

      # Set up state with demand
      state = %{state | in_progress_files: in_progress, file_timeout: 10_000, demand: 1}

      # Set up UUID to return new ID for re-enqueued file
      :meck.expect(UUID, :uuid4, fn -> "new-uuid" end)

      # Call handle_info with check_timeouts message
      {:noreply, dispatched, new_state} = FileQueueProducer.handle_info(:check_timeouts, state)

      # Verify timed out file was dispatched with new ID
      assert length(dispatched) == 1
      event = Enum.at(dispatched, 0)
      assert event.id == "new-uuid"
      assert event.path == file_path

      # Verify old file ID was removed from in_progress_files
      refute Map.has_key?(new_state.in_progress_files, file_id)

      # Verify new file ID is now in in_progress_files
      assert Map.has_key?(new_state.in_progress_files, "new-uuid")

      # Verify demand was used
      assert new_state.demand == 0
    end

    test "re-enqueues timed out files to events list when no demand", %{state: state} do
      # Current time for testing
      now = System.system_time(:millisecond)

      # Set up in_progress_files with a timed out file
      file_id = "timed-out"
      file_path = "/path/to/timeout.txt"
      metadata = %{size: 100, type: "text"}

      in_progress = %{
        file_id => %{
          path: file_path,
          metadata: metadata,
          # 15 seconds ago, over timeout
          enqueued_at: now - 15_000
        }
      }

      # Set up state with no demand
      state = %{state | in_progress_files: in_progress, file_timeout: 10_000, demand: 0}

      # Set up UUID to return new ID for re-enqueued file
      :meck.expect(UUID, :uuid4, fn -> "new-uuid" end)

      # Call handle_info with check_timeouts message
      {:noreply, [], new_state} = FileQueueProducer.handle_info(:check_timeouts, state)

      # Verify timed out file was added to events list with new ID
      assert length(new_state.events) == 1
      event = Enum.at(new_state.events, 0)
      assert event.id == "new-uuid"
      assert event.path == file_path

      # Verify old file ID was removed from in_progress_files
      refute Map.has_key?(new_state.in_progress_files, file_id)
    end
  end
end
