defmodule Pipeline.Utils.RetryTest do
  use ExUnit.Case, async: true

  alias Pipeline.Utils.Retry

  setup do
    # Start an agent to track call count across processes
    {:ok, agent} = Agent.start_link(fn -> 0 end)
    {:ok, %{agent: agent}}
  end

  # Instead of mocking, we'll just use a very small delay
  # and override the implementation for testing
  defmodule TestRetry do
    # Test implementation that matches the behavior of the real module
    # but without actual delays
    def retry_with_backoff(fun, operation_name \\ "operation", opts \\ []) do
      options = Keyword.merge([max_retries: 3, initial_delay: 1], opts)
      max_retries = options[:max_retries]

      do_retry(fun, operation_name, max_retries, 0)
    end

    defp do_retry(fun, operation_name, max_retries, current_retry) do
      try do
        case fun.() do
          {:ok, _} = result ->
            result

          {:error, _reason} = error ->
            if current_retry < max_retries do
              do_retry(fun, operation_name, max_retries, current_retry + 1)
            else
              {:error, error}
            end

          other ->
            other
        end
      rescue
        e ->
          if current_retry < max_retries do
            do_retry(fun, operation_name, max_retries, current_retry + 1)
          else
            {:error, e}
          end
      end
    end
  end

  describe "retry_with_backoff/3" do
    test "succeeds on first attempt" do
      fun = fn -> {:ok, "success on first try"} end
      result = Retry.retry_with_backoff(fun, "test operation")
      assert result == {:ok, "success on first try"}
    end

    test "retries on error tuple and succeeds" do
      counter = :counters.new(1, [])

      fun = fn ->
        count = :counters.get(counter, 1)
        :counters.add(counter, 1, 1)

        case count do
          0 -> {:error, "temporary error"}
          _ -> {:ok, "success after retry"}
        end
      end

      result = Retry.retry_with_backoff(fun, "retry test", max_retries: 3, initial_delay: 1)
      assert result == {:ok, "success after retry"}
      # One initial attempt + one retry = 2 calls
      assert :counters.get(counter, 1) == 2
    end

    test "gives up after max retries on error tuple" do
      counter = :counters.new(1, [])
      error_msg = "persistent error"

      fun = fn ->
        :counters.add(counter, 1, 1)
        {:error, error_msg}
      end

      result = Retry.retry_with_backoff(fun, "failing test", max_retries: 3, initial_delay: 1)
      # Original implementation wraps the error once
      assert result == {:error, {:error, error_msg}}
      # Initial attempt + 3 retries = 4 calls
      assert :counters.get(counter, 1) == 4
    end

    test "retries on exception and succeeds" do
      counter = :counters.new(1, [])

      fun = fn ->
        count = :counters.get(counter, 1)
        :counters.add(counter, 1, 1)

        case count do
          0 -> raise "temporary boom"
          _ -> {:ok, "success after exception"}
        end
      end

      result = Retry.retry_with_backoff(fun, "exception test", max_retries: 3, initial_delay: 1)
      assert result == {:ok, "success after exception"}
      # One initial attempt + one retry = 2 calls
      assert :counters.get(counter, 1) == 2
    end

    test "gives up after max retries on exception" do
      counter = :counters.new(1, [])

      fun = fn ->
        :counters.add(counter, 1, 1)
        raise "persistent boom"
      end

      result =
        Retry.retry_with_backoff(fun, "failing exception test", max_retries: 3, initial_delay: 1)

      assert {:error, %RuntimeError{message: "persistent boom"}} = result
      # Initial attempt + 3 retries = 4 calls
      assert :counters.get(counter, 1) == 4
    end
  end
end
