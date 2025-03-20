defmodule ConnectionHandler.Client do
  require Logger

  @max_retries 3
  @backoff_base 100  # milliseconds

  def command(command, opts \\ []) do
    retries = Keyword.get(opts, :retries, @max_retries)
    do_command(command, retries)
  end

  defp do_command(command, retries) when retries > 0 do
    case ConnectionHandler.PoolSupervisor.get_connection() do
      {:ok, conn} ->
        case Redix.command(conn, command) do
          {:ok, result} ->
            {:ok, result}
          {:error, %Redix.ConnectionError{} = error} ->
            # Only retry connection errors
            Logger.warning("Redis command failed with connection error: #{inspect(error)}")
            backoff = :math.pow(2, @max_retries - retries) * @backoff_base |> round()
            Process.sleep(backoff)
            do_command(command, retries - 1)
          other_error ->
            other_error
        end
      {:error, reason} ->
        Logger.error("Could not get Redis connection: #{inspect(reason)}")
        backoff = :math.pow(2, @max_retries - retries) * @backoff_base |> round()
        Process.sleep(backoff)
        do_command(command, retries - 1)
    end
  end

  defp do_command(command, 0) do
    Logger.error("Redis command #{inspect(command)} failed after maximum retries")
    {:error, :max_retries_exceeded}
  end

  # Helper functions for common Redis operations
  def get(key, opts \\ []), do: command(["GET", key], opts)
  def set(key, value, opts \\ []), do: command(["SET", key, value], opts)
  def sadd(set, member, opts \\ []), do: command(["SADD", set, member], opts)
  def smembers(set, opts \\ []), do: command(["SMEMBERS", set], opts)
  def del(key, opts \\ []), do: command(["DEL", key], opts)
end
