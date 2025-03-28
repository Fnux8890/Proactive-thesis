defmodule Pipeline.FaultHandling.Retry do
  @moduledoc """
  Implements retry policies for handling errors in pipeline stages.

  This module provides configurable retry strategies including:
  - Exponential backoff
  - Fixed interval retries
  - Maximum retry limits
  """

  # Milliseconds
  @default_base_delay 100
  # 30 seconds
  @default_max_delay 30_000
  @default_max_retries 5

  @doc """
  Initialize a new retry policy with optional configuration.

  ## Options
    * :base_delay - Base delay in milliseconds for first retry. Default: 100ms
    * :max_delay - Maximum delay between retries in milliseconds. Default: 30 seconds
    * :max_retries - Maximum number of retry attempts. Default: 5
    * :jitter - Boolean indicating whether to add random jitter to delay. Default: true
  """
  def new(opts \\ []) do
    %{
      base_delay: Keyword.get(opts, :base_delay, @default_base_delay),
      max_delay: Keyword.get(opts, :max_delay, @default_max_delay),
      max_retries: Keyword.get(opts, :max_retries, @default_max_retries),
      jitter: Keyword.get(opts, :jitter, true)
    }
  end

  @doc """
  Calculate backoff time in milliseconds based on retry count.

  Uses exponential backoff with optional jitter to prevent thundering herd problem.

  ## Parameters
    * retry_count - The current retry attempt number (starting from 1)
    * policy - Optional retry policy configuration

  ## Returns
    * Integer delay in milliseconds
  """
  def calculate_backoff(retry_count, policy \\ nil) do
    policy = policy || new()

    # Use exponential backoff: base_delay * 2^(retry_count - 1)
    delay = (policy.base_delay * :math.pow(2, retry_count - 1)) |> round()

    # Cap at max delay
    delay = min(delay, policy.max_delay)

    # Add jitter if enabled (±20% randomness)
    if policy.jitter do
      # Random value between 0.8 and 1.2
      jitter_factor = :rand.uniform() * 0.4 + 0.8
      round(delay * jitter_factor)
    else
      delay
    end
  end

  @doc """
  Determine if a retry should be attempted based on retry count and policy.

  ## Parameters
    * retry_count - The current retry attempt number
    * policy - Optional retry policy configuration

  ## Returns
    * true if retry should be attempted, false otherwise
  """
  def should_retry?(retry_count, policy \\ nil) do
    policy = policy || new()
    retry_count < policy.max_retries
  end

  @doc """
  Classify an error to determine if it's retryable.

  ## Parameters
    * error - The error or exception to classify

  ## Returns
    * :retryable - Error is transient and can be retried
    * :permanent - Error is permanent and should not be retried
    * :throttle - Error is due to rate limiting and should be retried with longer backoff
  """
  def classify_error(error) do
    cond do
      # Network-related errors are typically retryable
      is_connection_error?(error) -> :retryable
      # Rate limiting and resource exhaustion errors should be throttled
      is_throttle_error?(error) -> :throttle
      # Permission, validation errors are permanent
      is_permanent_error?(error) -> :permanent
      # By default, treat as retryable
      true -> :retryable
    end
  end

  # Determine if error is a connection-related error
  defp is_connection_error?(error) do
    cond do
      # For Redix errors
      match?(%Redix.ConnectionError{}, error) ->
        true

      # For DBConnection errors - check by struct name safely
      is_map(error) and Map.has_key?(error, :__struct__) and
          to_string(Map.get(error, :__struct__)) =~ "DBConnection.ConnectionError" ->
        true

      # For RuntimeError with connection-related messages
      match?(%RuntimeError{}, error) ->
        msg = error.message

        String.contains?(msg, "connection") ||
          String.contains?(msg, "timeout") ||
          String.contains?(msg, "network")

      # Default case
      true ->
        false
    end
  end

  # Determine if error is a throttling or rate-limiting error
  defp is_throttle_error?(error) do
    case error do
      %RuntimeError{message: msg} ->
        String.contains?(msg, "rate limit") ||
          String.contains?(msg, "too many requests") ||
          String.contains?(msg, "throttle")

      _ ->
        false
    end
  end

  # Determine if error is permanent (not retryable)
  defp is_permanent_error?(error) do
    case error do
      %ArgumentError{} ->
        true

      %RuntimeError{message: msg} ->
        String.contains?(msg, "permission denied") ||
          String.contains?(msg, "not found") ||
          String.contains?(msg, "invalid") ||
          String.contains?(msg, "permission")

      _ ->
        false
    end
  end
end
