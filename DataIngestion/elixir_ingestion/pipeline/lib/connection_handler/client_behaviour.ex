defmodule ConnectionHandler.ClientBehaviour do
  @moduledoc """
  Behaviour for Redis client operations.

  This module defines the interface for Redis client operations,
  which allows for dependency injection and testing with mocks.
  """

  @doc """
  Executes a Redis command.

  ## Parameters

    * `command` - The Redis command to execute, as a list of strings.

  ## Returns

  Returns one of:
    * `{:ok, result}` - The command was executed successfully.
    * `{:error, reason}` - An error occurred.
  """
  @callback command(command :: list(String.t())) :: {:ok, term()} | {:error, term()}

  @doc """
  Executes multiple Redis commands in a pipeline.

  ## Parameters

    * `commands` - A list of Redis commands, where each command is a list of strings.

  ## Returns

  Returns one of:
    * `{:ok, results}` - The pipeline was executed successfully, with results as a list.
    * `{:error, reason}` - An error occurred.
  """
  @callback pipeline(commands :: list(list(String.t()))) :: {:ok, list(term())} | {:error, term()}

  @doc """
  Gets a value from Redis.

  ## Parameters

    * `key` - The key to get.

  ## Returns

  Returns one of:
    * `{:ok, value}` - The value was retrieved successfully.
    * `{:error, reason}` - An error occurred.
  """
  @callback get(key :: String.t()) :: {:ok, term()} | {:error, term()}

  @doc """
  Sets a value in Redis.

  ## Parameters

    * `key` - The key to set.
    * `value` - The value to set.

  ## Returns

  Returns one of:
    * `{:ok, "OK"}` - The value was set successfully.
    * `{:error, reason}` - An error occurred.
  """
  @callback set(key :: String.t(), value :: String.t()) :: {:ok, String.t()} | {:error, term()}

  @doc """
  Deletes a key from Redis.

  ## Parameters

    * `key` - The key to delete.

  ## Returns

  Returns one of:
    * `{:ok, integer}` - The number of keys that were deleted.
    * `{:error, reason}` - An error occurred.
  """
  @callback del(key :: String.t()) :: {:ok, integer()} | {:error, term()}

  @doc """
  Checks if a member exists in a set.

  ## Parameters

    * `key` - The key of the set.
    * `member` - The member to check for.

  ## Returns

  Returns one of:
    * `{:ok, 1}` - The member exists in the set.
    * `{:ok, 0}` - The member does not exist in the set.
    * `{:error, reason}` - An error occurred.
  """
  @callback sismember(key :: String.t(), member :: String.t()) ::
              {:ok, integer()} | {:error, term()}

  @doc """
  Adds a member to a set.

  ## Parameters

    * `key` - The key of the set.
    * `member` - The member to add.

  ## Returns

  Returns one of:
    * `{:ok, 1}` - The member was added to the set.
    * `{:ok, 0}` - The member was already in the set.
    * `{:error, reason}` - An error occurred.
  """
  @callback sadd(key :: String.t(), member :: String.t()) :: {:ok, integer()} | {:error, term()}

  @doc """
  Removes a member from a set.

  ## Parameters

    * `key` - The key of the set.
    * `member` - The member to remove.

  ## Returns

  Returns one of:
    * `{:ok, 1}` - The member was removed from the set.
    * `{:ok, 0}` - The member was not in the set.
    * `{:error, reason}` - An error occurred.
  """
  @callback srem(key :: String.t(), member :: String.t()) :: {:ok, integer()} | {:error, term()}

  @doc """
  Gets all members of a set.

  ## Parameters

    * `key` - The key of the set.

  ## Returns

  Returns one of:
    * `{:ok, members}` - The members of the set, as a list.
    * `{:error, reason}` - An error occurred.
  """
  @callback smembers(key :: String.t()) :: {:ok, list(String.t())} | {:error, term()}

  @doc """
  Gets multiple values from Redis.

  ## Parameters

    * `keys` - The keys to get.

  ## Returns

  Returns one of:
    * `{:ok, values}` - The values were retrieved successfully, as a list.
    * `{:error, reason}` - An error occurred.
  """
  @callback mget(keys :: list(String.t())) :: {:ok, list(term())} | {:error, term()}

  @doc """
  Sets an expiration time (in seconds) on a key.

  ## Parameters

    * `key` - The key to set expiration on.
    * `seconds` - Time-to-live in seconds.

  ## Returns

  Returns one of:
    * `{:ok, 1}` - If the timeout was set.
    * `{:ok, 0}` - If the key does not exist.
    * `{:error, reason}` - If setting expiration failed.
  """
  @callback expire(key :: String.t(), seconds :: integer()) :: {:ok, integer()} | {:error, term()}
end
