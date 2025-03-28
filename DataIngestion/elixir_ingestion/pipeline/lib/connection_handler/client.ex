defmodule ConnectionHandler.Client do
  @moduledoc """
  Client for Redis commands with automatic retry functionality.
  This module provides a wrapper around Redis commands with retry logic.
  """
  @behaviour ConnectionHandler.ClientBehaviour

  require Logger
  alias ConnectionHandler.PoolSupervisor
  alias Pipeline.Utils.Retry

  @doc """
  Executes a Redis command.

  ## Parameters
    * command - The Redis command to execute as a list

  ## Returns
    * {:ok, result} - The result of the Redis command
    * {:error, reason} - If command failed after max retries
  """
  @impl true
  def command(command) when is_list(command) do
    Logger.debug("Executing Redis command: #{inspect(command)}")

    [cmd_name | args] = command
    execute_command(cmd_name, args)
  end

  # Legacy interface support
  def command(command, args) when is_binary(command) and is_list(args) do
    Logger.debug("Executing Redis command (legacy): #{command} #{inspect(args)}")
    execute_command(command, args)
  end

  @doc """
  Gets a value from Redis for the given key.

  ## Parameters
    * key - The key to retrieve

  ## Returns
    * {:ok, value} - The value for the key
    * {:ok, nil} - If the key does not exist
    * {:error, reason} - If getting the value failed
  """
  @impl true
  def get(key) do
    command(["GET", key])
  end

  @doc """
  Sets a key to the given value in Redis.

  ## Parameters
    * key - The key to set
    * value - The value to set

  ## Returns
    * {:ok, "OK"} - If the key was set
    * {:error, reason} - If setting the key failed
  """
  @impl true
  def set(key, value) do
    command(["SET", key, value])
  end

  @doc """
  Deletes a key from Redis.

  ## Parameters
    * key - The key to delete

  ## Returns
    * {:ok, count} - Number of keys deleted
    * {:error, reason} - If deletion failed
  """
  @impl true
  def del(key) do
    command(["DEL", key])
  end

  @doc """
  Checks if a member exists in a Redis set.

  ## Parameters
    * key - The set key
    * member - The member to check for

  ## Returns
    * {:ok, 1} - If the member exists in the set
    * {:ok, 0} - If the member does not exist in the set
    * {:error, reason} - If checking failed
  """
  @impl true
  def sismember(key, member) do
    command(["SISMEMBER", key, member])
  end

  @doc """
  Adds a member to a Redis set.

  ## Parameters
    * key - The set key
    * member - The member to add

  ## Returns
    * {:ok, 1} - If the member was added
    * {:ok, 0} - If the member was already in the set
    * {:error, reason} - If adding failed
  """
  @impl true
  def sadd(key, member) do
    command(["SADD", key, member])
  end

  @doc """
  Removes a member from a Redis set.

  ## Parameters
    * key - The set key
    * member - The member to remove

  ## Returns
    * {:ok, 1} - If the member was removed
    * {:ok, 0} - If the member was not in the set
    * {:error, reason} - If removal failed
  """
  @impl true
  def srem(key, member) do
    command(["SREM", key, member])
  end

  @doc """
  Gets all members of a Redis set.

  ## Parameters
    * key - The set key

  ## Returns
    * {:ok, members} - List of members
    * {:error, reason} - If getting members failed
  """
  @impl true
  def smembers(key) do
    command(["SMEMBERS", key])
  end

  @doc """
  Gets multiple values from Redis.

  ## Parameters
    * keys - List of keys to retrieve

  ## Returns
    * {:ok, values} - List of values
    * {:error, reason} - If getting values failed
  """
  @impl true
  def mget(keys) do
    command(["MGET" | keys])
  end

  @doc """
  Starts a Redis transaction.

  ## Returns
    * {:ok, "OK"} - If transaction was started
    * {:error, reason} - If starting transaction failed
  """
  def multi do
    command(["MULTI"])
  end

  @doc """
  Executes a Redis transaction.

  ## Returns
    * {:ok, results} - Results of transaction commands
    * {:error, reason} - If executing transaction failed
  """
  def exec do
    command(["EXEC"])
  end

  @doc """
  Executes a Redis pipeline (multiple commands sent at once).

  ## Parameters
    * commands - List of command lists

  ## Returns
    * {:ok, results} - List of results
    * {:error, reason} - If pipeline failed
  """
  @impl true
  def pipeline(commands) do
    Retry.retry_with_backoff(
      fn ->
        case PoolSupervisor.get_connection() do
          {:ok, conn} ->
            try do
              case Redix.pipeline(conn, commands) do
                {:ok, results} ->
                  {:ok, results}

                {:error, %{reason: :closed}} ->
                  Logger.warning("Redis connection closed during pipeline, retrying...")
                  notify_connection_failure(conn)
                  {:error, :closed}

                {:error, reason} = error ->
                  Logger.error("Redis pipeline failed: #{inspect(reason)}")
                  error
              end
            rescue
              e ->
                Logger.error("Exception during Redis pipeline: #{inspect(e)}")
                notify_connection_failure(conn)
                {:error, {:exception, e}}
            end

          {:error, reason} ->
            Logger.error("Failed to get Redis connection: #{inspect(reason)}")
            {:error, reason}
        end
      end,
      "Redis pipeline",
      max_retries: 3,
      initial_delay: 100
    )
  end

  @doc """
  Sets an expiration time (in seconds) on a key.

  ## Parameters
    * key - The key to set expiration on
    * seconds - Time-to-live in seconds

  ## Returns
    * {:ok, 1} - If the timeout was set
    * {:ok, 0} - If the key does not exist
    * {:error, reason} - If setting expiration failed
  """
  @impl true
  def expire(key, seconds) when is_binary(key) and is_integer(seconds) do
    command(["EXPIRE", key, to_string(seconds)])
  end

  # Notify the pool supervisor about a connection failure if the function exists
  defp notify_connection_failure(conn) do
    # Check if the handle_connection_failure function exists in the PoolSupervisor module
    if function_exported?(PoolSupervisor, :handle_connection_failure, 1) do
      PoolSupervisor.handle_connection_failure(conn)
    else
      Logger.warning(
        "PoolSupervisor.handle_connection_failure/1 not implemented, connection failure not handled"
      )
    end
  end

  # Private functions

  defp execute_command(command, args) do
    Retry.retry_with_backoff(
      fn ->
        case PoolSupervisor.get_connection() do
          {:ok, conn} ->
            try do
              case Redix.command(conn, [command | args]) do
                {:ok, result} ->
                  {:ok, result}

                {:error, %{reason: :closed}} ->
                  Logger.warning("Redis connection closed, retrying...")
                  notify_connection_failure(conn)
                  {:error, :closed}

                {:error, reason} = error ->
                  Logger.error("Redis command failed: #{inspect(reason)}")
                  error
              end
            rescue
              e ->
                Logger.error("Exception during Redis command: #{inspect(e)}")
                notify_connection_failure(conn)
                {:error, {:exception, e}}
            end

          {:error, reason} ->
            Logger.error("Failed to get Redis connection: #{inspect(reason)}")
            {:error, reason}
        end
      end,
      "Redis command #{command}",
      max_retries: 3,
      initial_delay: 100
    )
  end
end
