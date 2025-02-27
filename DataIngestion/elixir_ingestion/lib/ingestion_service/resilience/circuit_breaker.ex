defmodule IngestionService.Resilience.CircuitBreaker do
  use GenServer
  require Logger

  @moduledoc """
  Implements the Circuit Breaker pattern to prevent cascading failures
  and provide more resilient error handling.

  The circuit breaker has three states:

  1. CLOSED: All requests pass through normally. If failures exceed the threshold,
     the circuit transitions to OPEN.

  2. OPEN: Requests fail fast without executing. After the reset timeout,
     the circuit transitions to HALF_OPEN.

  3. HALF_OPEN: Allows a limited number of test requests. If successful,
     the circuit transitions to CLOSED. If they fail, it returns to OPEN.
  """

  # Default configuration
  @default_failure_threshold 5
  # milliseconds
  @default_reset_timeout 30_000
  @default_half_open_max 1

  # Client API

  @doc """
  Starts the circuit breaker registry server.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Executes a function with circuit breaker protection.

  ## Options

  * `:failure_threshold` - Number of failures before opening the circuit (default: 5)
  * `:reset_timeout` - Time in milliseconds before attempting recovery (default: 30000)
  * `:half_open_max` - Maximum number of test requests in HALF_OPEN state (default: 1)
  """
  def execute(circuit_name, fun, opts \\ []) do
    case get_circuit_state(circuit_name) do
      :open ->
        # Circuit is open, fail fast
        Logger.warn("Circuit #{circuit_name} is OPEN, failing fast")
        {:error, :circuit_open}

      :half_open ->
        # Test the circuit with one request
        Logger.info("Circuit #{circuit_name} is HALF_OPEN, testing recovery")
        try_circuit_recovery(circuit_name, fun, opts)

      :closed ->
        # Normal operation
        protected_execute(circuit_name, fun, opts)
    end
  end

  @doc """
  Gets the current state of a circuit.
  """
  def get_circuit_state(circuit_name) do
    case GenServer.call(__MODULE__, {:get_circuit, circuit_name}) do
      {:ok, circuit} -> circuit.state
      # Default to closed if circuit doesn't exist
      :error -> :closed
    end
  end

  @doc """
  Manually resets a circuit to the closed state.
  """
  def reset_circuit(circuit_name) do
    GenServer.call(__MODULE__, {:reset_circuit, circuit_name})
  end

  @doc """
  Lists all circuits and their current states.
  """
  def list_circuits do
    GenServer.call(__MODULE__, :list_circuits)
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    Logger.info("Starting circuit breaker registry")
    {:ok, %{circuits: %{}}}
  end

  @impl true
  def handle_call({:get_circuit, name}, _from, state) do
    case Map.get(state.circuits, name) do
      nil -> {:reply, :error, state}
      circuit -> {:reply, {:ok, circuit}, state}
    end
  end

  @impl true
  def handle_call({:reset_circuit, name}, _from, state) do
    new_state =
      case Map.get(state.circuits, name) do
        nil ->
          # Circuit doesn't exist, do nothing
          state

        _circuit ->
          # Reset the circuit to closed state
          new_circuit = %{
            state: :closed,
            failure_count: 0,
            last_failure: nil,
            last_success: System.system_time(:millisecond),
            opened_at: nil
          }

          new_circuits = Map.put(state.circuits, name, new_circuit)
          %{state | circuits: new_circuits}
      end

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:list_circuits, _from, state) do
    circuit_list =
      state.circuits
      |> Enum.map(fn {name, circuit} ->
        {name, circuit.state, circuit.failure_count}
      end)

    {:reply, circuit_list, state}
  end

  @impl true
  def handle_call({:record_success, name}, _from, state) do
    new_state =
      case Map.get(state.circuits, name) do
        nil ->
          # Circuit doesn't exist, create it with success
          new_circuit = %{
            state: :closed,
            failure_count: 0,
            last_failure: nil,
            last_success: System.system_time(:millisecond),
            opened_at: nil
          }

          new_circuits = Map.put(state.circuits, name, new_circuit)
          %{state | circuits: new_circuits}

        circuit ->
          # Update existing circuit
          new_circuit =
            case circuit.state do
              :half_open ->
                # Success in half-open state transitions back to closed
                Logger.info(
                  "Circuit #{name} recovery successful, transitioning from HALF_OPEN to CLOSED"
                )

                %{
                  state: :closed,
                  failure_count: 0,
                  last_failure: circuit.last_failure,
                  last_success: System.system_time(:millisecond),
                  opened_at: nil
                }

              _ ->
                # Success in other states just updates the stats
                %{
                  state: :closed,
                  failure_count: 0,
                  last_failure: circuit.last_failure,
                  last_success: System.system_time(:millisecond),
                  opened_at: circuit.opened_at
                }
            end

          new_circuits = Map.put(state.circuits, name, new_circuit)
          %{state | circuits: new_circuits}
      end

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:record_failure, name, options}, _from, state) do
    # Get thresholds from options
    failure_threshold = Keyword.get(options, :failure_threshold, @default_failure_threshold)
    reset_timeout = Keyword.get(options, :reset_timeout, @default_reset_timeout)

    new_state =
      case Map.get(state.circuits, name) do
        nil ->
          # Circuit doesn't exist, create it with failure
          new_circuit = %{
            state: :closed,
            failure_count: 1,
            last_failure: System.system_time(:millisecond),
            last_success: nil,
            opened_at: nil
          }

          new_circuits = Map.put(state.circuits, name, new_circuit)
          %{state | circuits: new_circuits}

        circuit ->
          current_time = System.system_time(:millisecond)

          new_circuit =
            case circuit.state do
              :open ->
                # Already open, just update the failure time
                %{circuit | last_failure: current_time}

              :half_open ->
                # Failure in half-open means we go back to open
                Logger.warn("Circuit #{name} recovery failed, resetting to OPEN")

                %{
                  state: :open,
                  failure_count: circuit.failure_count + 1,
                  last_failure: current_time,
                  last_success: circuit.last_success,
                  opened_at: current_time,
                  reset_after: current_time + reset_timeout
                }

              :closed ->
                # In closed state, increment failure count
                new_failure_count = circuit.failure_count + 1

                if new_failure_count >= failure_threshold do
                  # Too many failures, open the circuit
                  Logger.warn(
                    "Circuit #{name} exceeded failure threshold (#{new_failure_count}/#{failure_threshold}), opening circuit"
                  )

                  %{
                    state: :open,
                    failure_count: new_failure_count,
                    last_failure: current_time,
                    last_success: circuit.last_success,
                    opened_at: current_time,
                    reset_after: current_time + reset_timeout
                  }
                else
                  # Not enough failures yet, just update the count
                  %{circuit | failure_count: new_failure_count, last_failure: current_time}
                end
            end

          new_circuits = Map.put(state.circuits, name, new_circuit)
          %{state | circuits: new_circuits}
      end

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_info({:check_circuits}, state) do
    # Check for circuits that should transition from open to half-open
    current_time = System.system_time(:millisecond)

    new_circuits =
      state.circuits
      |> Enum.map(fn {name, circuit} ->
        case circuit do
          %{state: :open, reset_after: reset_time} when reset_time <= current_time ->
            # Transition to half-open
            Logger.info("Circuit #{name} timeout elapsed, transitioning from OPEN to HALF_OPEN")
            {name, %{circuit | state: :half_open}}

          _ ->
            # No change
            {name, circuit}
        end
      end)
      |> Map.new()

    # Schedule next check
    schedule_circuit_check()

    {:noreply, %{state | circuits: new_circuits}}
  end

  # Private functions

  # Executes with circuit breaker protection
  defp protected_execute(circuit_name, fun, opts) do
    try do
      # Call the function
      result = fun.()

      # Record success
      record_success(circuit_name)

      # Return the result
      {:ok, result}
    rescue
      e ->
        # Record failure
        record_failure(circuit_name, e, opts)

        # Return error
        {:error, e}
    end
  end

  # Attempts recovery in half-open state
  defp try_circuit_recovery(circuit_name, fun, opts) do
    # Try to execute and see if it succeeds
    case protected_execute(circuit_name, fun, opts) do
      {:ok, result} ->
        # Recovery successful
        {:ok, result}

      {:error, error} ->
        # Recovery failed
        {:error, error}
    end
  end

  # Records a successful execution
  defp record_success(circuit_name) do
    GenServer.call(__MODULE__, {:record_success, circuit_name})
  end

  # Records a failed execution
  defp record_failure(circuit_name, error, opts) do
    Logger.error("Circuit #{circuit_name} experienced failure: #{inspect(error)}")
    GenServer.call(__MODULE__, {:record_failure, circuit_name, opts})
  end

  # Schedules periodic circuit check
  defp schedule_circuit_check do
    # Check circuits every 5 seconds
    Process.send_after(self(), {:check_circuits}, 5000)
  end
end
