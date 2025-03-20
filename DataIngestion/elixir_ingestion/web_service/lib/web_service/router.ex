defmodule WebService.Router do
  use Plug.Router
  use Plug.ErrorHandler

  alias WebService.ResultsMonitor

  # Required Plug Router setup
  plug Plug.Logger
  plug :match
  plug Plug.Parsers,
    parsers: [:json, :urlencoded, :multipart],
    pass: ["*/*"],
    json_decoder: Jason
  plug :dispatch

  # Health check endpoint for Docker
  get "/health" do
    send_resp(conn, 200, Jason.encode!(%{status: "ok"}))
  end

  # Main dashboard page
  get "/" do
    send_resp(conn, 200, "Data Ingestion Pipeline - Dashboard")
  end

  # API endpoints for data pipeline management
  get "/api/status" do
    results = ResultsMonitor.get_results()
    send_resp(conn, 200, Jason.encode!(%{status: "ok", results: results}))
  end

  # Process new data files
  post "/api/process" do
    case conn.body_params do
      %{"file" => file_path} ->
        # Example processing - you would implement the actual pipeline trigger here
        Registry.register(WebService.ProcessingRegistry, file_path, %{status: "processing"})
        send_resp(conn, 200, Jason.encode!(%{status: "processing", file: file_path}))
      _ ->
        send_resp(conn, 400, Jason.encode!(%{error: "Invalid request. Must include 'file' parameter."}))
    end
  end

  # View specific result
  get "/api/results/:file" do
    case Registry.lookup(WebService.ProcessingRegistry, file) do
      [{_, status}] ->
        send_resp(conn, 200, Jason.encode!(%{file: file, status: status}))
      [] ->
        send_resp(conn, 404, Jason.encode!(%{error: "File not found"}))
    end
  end

  # Catch-all route
  match _ do
    send_resp(conn, 404, "Not Found")
  end

  # Error handler implementation
  def handle_errors(conn, %{kind: kind, reason: reason, stack: stack}) do
    IO.inspect(kind, label: "Error kind")
    IO.inspect(reason, label: "Error reason")
    IO.inspect(stack, label: "Error stack")

    send_resp(conn, conn.status || 500, "Something went wrong")
  end
end
