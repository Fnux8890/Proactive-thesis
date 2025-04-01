defmodule ConnectionHandler.ClientTest do
  use ExUnit.Case, async: false

  alias ConnectionHandler.Client

  test "pipeline with mocked implementation" do
    # Just test pipeline directly with mocked dependencies
    # Don't try to mock the Client's internal functions

    # Mock the pool supervisor to return a connection
    :meck.new(ConnectionHandler.PoolSupervisor, [:passthrough])

    :meck.expect(ConnectionHandler.PoolSupervisor, :get_connection, fn ->
      {:ok, :mock_connection}
    end)

    # Mock Redix to return a known value
    :meck.new(Redix, [:passthrough])

    :meck.expect(Redix, :pipeline, fn _conn, commands ->
      # Verify the commands being sent to pipeline
      assert length(commands) == 1
      assert List.first(commands) == ["GET", "test_key"]

      # Return a successful result
      {:ok, ["mock_value"]}
    end)

    # Call the function under test
    result = Client.pipeline([["GET", "test_key"]])

    # Verify the result
    assert result == {:ok, ["mock_value"]}

    # Clean up mocks
    :meck.unload(ConnectionHandler.PoolSupervisor)
    :meck.unload(Redix)
  end

  test "get with list command style" do
    # Mock the pool supervisor to return a connection
    :meck.new(ConnectionHandler.PoolSupervisor, [:passthrough])

    :meck.expect(ConnectionHandler.PoolSupervisor, :get_connection, fn ->
      {:ok, :mock_connection}
    end)

    # Mock Redix to return a known value
    :meck.new(Redix, [:passthrough])

    :meck.expect(Redix, :command, fn _conn, cmd ->
      # Verify the command being sent
      assert cmd == ["GET", "test_key"]

      # Return a successful result
      {:ok, "mock_value"}
    end)

    # Call the function under test with list style
    # This directly calls command/1 which should be simpler to test
    result = Client.command(["GET", "test_key"])

    # Verify the result
    assert result == {:ok, "mock_value"}

    # Clean up mocks
    :meck.unload(ConnectionHandler.PoolSupervisor)
    :meck.unload(Redix)
  end
end
