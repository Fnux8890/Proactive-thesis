ExUnit.start()

# Define the mock behavior
defmodule ConnectionHandler.ClientBehavior do
  @callback smembers(String.t()) :: {:ok, list(String.t())} | {:error, any()}
  @callback mget(list(String.t())) :: {:ok, list(String.t())} | {:error, any()}
  @callback pipeline(list(list(String.t()))) :: {:ok, list(any())} | {:error, any()}
  @callback sismember(String.t(), String.t()) :: {:ok, integer()} | {:error, any()}
end

# Create the mock with Mox
Mox.defmock(MockRedisClient, for: ConnectionHandler.ClientBehavior)
Mox.defmock(MockStateStore, for: FileWatcher.StateStoreBehaviour)
Mox.defmock(MockProducer, for: Producer.ProducerBehaviour)
Mox.defmock(MockFileSystem, for: FileWatcher.FileSystemBehaviour)
Mox.defmock(MockServer, for: FileWatcher.ServerBehaviour)

# Note: Each test will be responsible for:
# 1. Setting up verify_on_exit! in their own setup block
# 2. Allowing the mock for their test process via Mox.allow(MockX, self())
# 3. Setting up expectations or stubs as needed for their specific test cases
