Logger.error("File processing failed: #{file_path} by slot #{slot_id}. Reason: #{inspect(reason)}")
Pipeline.Tracking.track(file_path, :dispatcher_failure)

# Update state in Redis to permanently_failed
case RedisConnection.command(["HSET", @state_key, file_path, "permanently_failed"]) do
  {:ok, _} ->
    Logger.debug("Successfully updated Redis state to 'permanently_failed' for #{file_path}")

  {:error, err} ->
    Logger.error("Failed to update Redis state to 'permanently_failed' for #{file_path}: #{inspect(err)}")
end

# Remove processor info and add slot back to available
