defmodule Pipeline.Utils.Retry do
  @moduledoc """
  Provides retry functionality with exponential backoff.

  This module offers different strategies for retrying operations that may fail transiently.
  """

  require Logger

  @default_options [
    max_retries: 3,
    initial_delay: 100,
    max_delay: 30_000
  ]

  @doc """
  Retries a function with exponential backoff. Returns the result of the function or an error.

  ## Parameters

  * `fun` - The function to retry
  * `operation_name` - Name of the operation for logging
  * `opts` - Options for retry behavior
    * `:max_retries` - Maximum number of retries (default: 3)
    * `:initial_delay` - Initial delay in milliseconds (default: 100)
    * `:max_delay` - Maximum delay in milliseconds (default: 30_000)

  ## Examples

      iex> Retry.retry_with_backoff(fn -> {:ok, "success"} end, "test")
      {:ok, "success"}

      iex> Retry.retry_with_backoff(fn -> raise "error" end, "test", max_retries: 2)
      {:error, %RuntimeError{message: "error"}}
  """
  def retry_with_backoff(fun, operation_name \\ "operation", opts \\ [])

  # Handle the new style with options
  def retry_with_backoff(fun, operation_name, opts) when is_binary(operation_name) do
    options = Keyword.merge(@default_options, opts)
    max_retries = options[:max_retries]
    initial_delay = options[:initial_delay]
    max_delay = options[:max_delay]

    retry_with_backoff_internal(fun, operation_name, max_retries, initial_delay, max_delay, 0)
  end

  # For backward compatibility with the old style
  def retry_with_backoff(fun, max_retries, initial_delay)
      when is_integer(max_retries) and is_integer(initial_delay) do
    retry_with_backoff(fun, "operation", max_retries: max_retries, initial_delay: initial_delay)
  end

  @doc """
  Simpler retry mechanism with fixed delay. Returns the result of the function or an error.

  ## Parameters

  * `fun` - The function to retry
  * `operation_name` - Name of the operation for logging
  * `opts` - Options for retry behavior
    * `:retries` - Number of retries (default: 3)
    * `:delay` - Delay in milliseconds between retries (default: 100)

  ## Examples

      iex> Retry.retry(fn -> {:ok, "success"} end, "test")
      {:ok, "success"}

      iex> Retry.retry(fn -> {:error, "fail"} end, "test", retries: 2)
      {:error, "fail"}
  """
  def retry(fun, operation_name \\ "operation", opts \\ []) do
    retries = Keyword.get(opts, :retries, 3)
    delay = Keyword.get(opts, :delay, 100)

    retry_internal(fun, operation_name, retries, delay, 0)
  end

  # Private functions

  defp retry_with_backoff_internal(
         fun,
         operation_name,
         max_retries,
         initial_delay,
         max_delay,
         current_retry
       ) do
    try do
      case fun.() do
        # Success case - return directly
        {:ok, _} = result ->
          result

        # Error tuple case - handle failure
        {:error, _reason} = error ->
          if current_retry < max_retries do
            backoff_time = calculate_backoff(initial_delay, current_retry, max_delay)

            Logger.debug(
              "Retrying #{operation_name} after error (#{current_retry + 1}/#{max_retries}) in #{backoff_time}ms: #{inspect(error)}"
            )

            Process.sleep(backoff_time)
            # Recurse for retry
            retry_with_backoff_internal(
              fun,
              operation_name,
              max_retries,
              initial_delay,
              max_delay,
              current_retry + 1
            )
          else
            # Max retries reached
            Logger.error(
              "Max retries (#{max_retries}) reached for #{operation_name}. Last error: #{inspect(error)}"
            )

            # Return the last error reason wrapped in an error tuple
            {:error, error}
          end

        # Pass through any other result
        other ->
          other
      end
    rescue
      exception ->
        if current_retry < max_retries do
          backoff_time = calculate_backoff(initial_delay, current_retry, max_delay)

          Logger.debug(
            "Retrying #{operation_name} after exception (#{current_retry + 1}/#{max_retries}) in #{backoff_time}ms: #{inspect(exception)}"
          )

          Process.sleep(backoff_time)
          # Recurse for retry
          retry_with_backoff_internal(
            fun,
            operation_name,
            max_retries,
            initial_delay,
            max_delay,
            current_retry + 1
          )
        else
          # Max retries reached, re-raise the exception
          Logger.error(
            "Max retries (#{max_retries}) reached for #{operation_name}. Re-raising exception: #{inspect(exception)}"
          )

          reraise exception, __STACKTRACE__
        end
    end
  end

  defp calculate_backoff(initial_delay, retry_count, max_delay) do
    min(initial_delay * :math.pow(2, retry_count), max_delay) |> trunc()
  end

  defp retry_internal(fun, operation_name, max_retries, delay, current_retry) do
    case fun.() do
      {:ok, _} = result ->
        result

      {:error, reason} = error ->
        if current_retry < max_retries do
          Logger.debug("Retrying #{operation_name} after error: #{inspect(reason)}")
          Process.sleep(delay)
          retry_internal(fun, operation_name, max_retries, delay, current_retry + 1)
        else
          Logger.debug(
            "Max retries (#{max_retries}) reached for #{operation_name}: #{inspect(reason)}"
          )

          error
        end

      other ->
        other
    end
  end
end
