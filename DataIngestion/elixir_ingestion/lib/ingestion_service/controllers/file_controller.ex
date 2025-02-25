defmodule IngestionService.FileController do
  use Phoenix.Controller, namespace: IngestionService
  
  require Logger
  
  @doc """
  Process a file by submitting it to the ingestion pipeline.
  Expects 'path' parameter with absolute file path, or file upload.
  """
  def process(conn, %{"path" => file_path} = params) do
    # Extract optional parameters
    priority = Map.get(params, "priority", "normal")
    source = Map.get(params, "source", determine_source(file_path))
    
    # Verify file exists
    if File.exists?(file_path) do
      # Submit file to pipeline for processing
      Logger.info("Manually processing file: #{file_path}")
      
      # Create tracking ID for the request
      tracking_id = "manual_#{:erlang.system_time(:millisecond)}_#{:rand.uniform(1000)}"
      
      # Store tracking info in Redis
      Redix.command(:redix, [
        "HSET", 
        "ingestion:tracking:#{tracking_id}", 
        "file_path", file_path,
        "status", "pending",
        "created_at", DateTime.utc_now() |> DateTime.to_iso8601(),
        "source", source,
        "priority", priority
      ])
      
      # Set TTL for tracking info (24 hours)
      Redix.command(:redix, ["EXPIRE", "ingestion:tracking:#{tracking_id}", 86400])
      
      # Notify the producer
      IngestionService.Pipeline.Producer.notify_file_change(file_path)
      
      json(conn, %{
        status: :ok,
        message: "File submitted for processing",
        tracking_id: tracking_id
      })
    else
      conn
      |> put_status(400)
      |> json(%{
        status: :error,
        message: "File not found: #{file_path}"
      })
    end
  end
  
  @doc """
  Process an uploaded file by storing it and submitting to the pipeline.
  """
  def process(conn, %{"file" => upload} = params) do
    # Create a temporary file from the upload
    upload_path = Path.join(System.tmp_dir!(), upload.filename)
    
    # Save the file
    File.cp!(upload.path, upload_path)
    
    # Extract optional parameters
    source = Map.get(params, "source", "api_upload")
    priority = Map.get(params, "priority", "high")
    
    # Create tracking ID
    tracking_id = "upload_#{:erlang.system_time(:millisecond)}_#{:rand.uniform(1000)}"
    
    # Store tracking info in Redis
    Redix.command(:redix, [
      "HSET", 
      "ingestion:tracking:#{tracking_id}", 
      "file_path", upload_path,
      "status", "pending",
      "created_at", DateTime.utc_now() |> DateTime.to_iso8601(),
      "source", source,
      "priority", priority,
      "original_filename", upload.filename
    ])
    
    # Set TTL for tracking info (24 hours)
    Redix.command(:redix, ["EXPIRE", "ingestion:tracking:#{tracking_id}", 86400])
    
    # Notify the producer
    IngestionService.Pipeline.Producer.notify_file_change(upload_path)
    
    json(conn, %{
      status: :ok,
      message: "File uploaded and submitted for processing",
      tracking_id: tracking_id
    })
  end
  
  def process(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{
      status: :error,
      message: "Missing required parameters. Please provide 'path' or upload a file."
    })
  end
  
  @doc """
  Check the status of a previously submitted file.
  """
  def status(conn, %{"id" => tracking_id}) do
    # Get tracking info from Redis
    case Redix.command(:redix, ["HGETALL", "ingestion:tracking:#{tracking_id}"]) do
      {:ok, []} ->
        # No tracking info found
        conn
        |> put_status(404)
        |> json(%{
          status: :error,
          message: "Tracking ID not found"
        })
        
      {:ok, tracking_info} ->
        # Convert list to map
        tracking_map = tracking_info
        |> Enum.chunk_every(2)
        |> Enum.map(fn [k, v] -> {k, v} end)
        |> Enum.into(%{})
        
        # Check if we have a file path
        file_path = Map.get(tracking_map, "file_path")
        
        # Check for results in the database
        results = 
          if file_path do
            query_db_for_results(file_path)
          else
            []
          end
        
        # Update status based on results
        status = 
          cond do
            Enum.any?(results, &(&1["success"] == true)) -> "completed"
            Enum.any?(results, &(&1["success"] == false)) -> "failed"
            true -> Map.get(tracking_map, "status", "pending")
          end
        
        # If status changed, update Redis
        if status != Map.get(tracking_map, "status") do
          Redix.command(:redix, [
            "HSET", 
            "ingestion:tracking:#{tracking_id}", 
            "status", status,
            "updated_at", DateTime.utc_now() |> DateTime.to_iso8601()
          ])
        end
        
        json(conn, %{
          tracking_id: tracking_id,
          status: status,
          file_path: file_path,
          created_at: Map.get(tracking_map, "created_at"),
          updated_at: Map.get(tracking_map, "updated_at"),
          results: results
        })
    end
  end
  
  @doc """
  List recently processed files.
  """
  def list(conn, params) do
    # Get limit from parameters
    limit = Map.get(params, "limit", "50") |> String.to_integer()
    source = Map.get(params, "source")
    
    # Build query based on parameters
    query = "SELECT * FROM ingestion_results ORDER BY processed_at DESC LIMIT $1"
    query_params = [limit]
    
    {query, query_params} = if source do
      {"SELECT * FROM ingestion_results WHERE source = $2 ORDER BY processed_at DESC LIMIT $1", [limit, source]}
    else
      {query, query_params}
    end
    
    # Query the database
    {:ok, result} = IngestionService.Repo.query(query, query_params)
    
    # Format the results
    files = result.rows
    |> Enum.map(fn row ->
      Enum.zip(result.columns, row) |> Enum.into(%{})
    end)
    
    json(conn, %{
      files: files,
      count: length(files)
    })
  end
  
  # Helper to determine source from file path
  defp determine_source(file_path) do
    path_parts = String.split(file_path, "/")
    cond do
      Enum.member?(path_parts, "aarslev") -> "aarslev"
      Enum.member?(path_parts, "knudjepsen") -> "knudjepsen"
      true -> "unknown"
    end
  end
  
  # Query the database for results
  defp query_db_for_results(file_path) do
    case IngestionService.Repo.query(
      "SELECT * FROM ingestion_results WHERE file_path = $1 ORDER BY processed_at DESC LIMIT 5",
      [file_path]
    ) do
      {:ok, result} ->
        # Format the results
        result.rows
        |> Enum.map(fn row ->
          Enum.zip(result.columns, row) |> Enum.into(%{})
        end)
        
      {:error, _} ->
        []
    end
  end
end 