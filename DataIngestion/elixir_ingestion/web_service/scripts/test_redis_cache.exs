# Test Redis cache script
# Run with: mix run scripts/test_redis_cache.exs

Mix.install([:redix])

defmodule RedisCacheTester do
  @moduledoc """
  A simple utility to test Redis connectivity and cache operations
  """
  
  def run do
    IO.puts("Testing Redis connection...")
    
    case connect_to_redis() do
      {:ok, conn} ->
        IO.puts("✅ Successfully connected to Redis")
        run_test_operations(conn)
        :ok
      {:error, reason} ->
        IO.puts("❌ Failed to connect to Redis: #{inspect(reason)}")
        :error
    end
  end
  
  defp connect_to_redis do
    redis_url = System.get_env("REDIS_URL") || "redis://localhost:6379"
    IO.puts("Connecting to Redis at #{redis_url}")
    Redix.start_link(redis_url)
  end
  
  defp run_test_operations(conn) do
    # Test basic operations
    test_key = "test:key:#{:os.system_time(:millisecond)}"
    test_value = "Hello from Elixir: #{DateTime.utc_now()}"
    
    IO.puts("\nTesting SET operation...")
    case Redix.command(conn, ["SET", test_key, test_value]) do
      {:ok, "OK"} ->
        IO.puts("✅ Successfully set value")
      {:error, reason} ->
        IO.puts("❌ Failed to set value: #{inspect(reason)}")
    end
    
    IO.puts("\nTesting GET operation...")
    case Redix.command(conn, ["GET", test_key]) do
      {:ok, ^test_value} ->
        IO.puts("✅ Successfully retrieved value: #{test_value}")
      {:ok, different_value} ->
        IO.puts("⚠️ Retrieved value doesn't match: #{different_value}")
      {:error, reason} ->
        IO.puts("❌ Failed to get value: #{inspect(reason)}")
    end
    
    IO.puts("\nTesting DEL operation...")
    case Redix.command(conn, ["DEL", test_key]) do
      {:ok, 1} ->
        IO.puts("✅ Successfully deleted key")
      {:ok, 0} ->
        IO.puts("⚠️ Key didn't exist")
      {:error, reason} ->
        IO.puts("❌ Failed to delete key: #{inspect(reason)}")
    end
    
    IO.puts("\nTesting connection to redis_cache instance...")
    case Redix.command(conn, ["PING"]) do
      {:ok, "PONG"} ->
        IO.puts("✅ Redis instance is responsive")
      {:error, reason} ->
        IO.puts("❌ Redis instance is not responsive: #{inspect(reason)}")
    end
  end
end

# Run the test
RedisCacheTester.run()
