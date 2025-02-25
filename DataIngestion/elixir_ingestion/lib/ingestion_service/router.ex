defmodule IngestionService.Router do
  use Phoenix.Router

  import Plug.Conn
  import Phoenix.Controller

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/api", IngestionService do
    pipe_through :api
    
    # Ingestion API endpoints
    get "/status", StatusController, :status
    get "/health", StatusController, :health
    get "/metrics", MetricsController, :index
    get "/metrics/latest", MetricsController, :latest
    
    # File processing endpoints
    post "/files/process", FileController, :process
    get "/files/status/:id", FileController, :status
    get "/files/list", FileController, :list
    
    # Pipeline management
    get "/pipeline/status", PipelineController, :status
    post "/pipeline/reload", PipelineController, :reload
    post "/pipeline/purge", PipelineController, :purge
    post "/pipeline/configure", PipelineController, :configure
  end

  # Enable LiveDashboard in development
  if Mix.env() in [:dev, :test] do
    import Phoenix.LiveDashboard.Router

    scope "/" do
      pipe_through [:fetch_session, :protect_from_forgery]
      live_dashboard "/dashboard", metrics: IngestionService.Telemetry
    end
  end

  # Enables the Swoosh mailbox preview in development.
  if Mix.env() == :dev do
    scope "/dev" do
      pipe_through [:fetch_session, :protect_from_forgery]

      forward "/mailbox", Plug.Swoosh.MailboxPreview
    end
  end
end 