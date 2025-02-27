defmodule IngestionServiceWeb.Router do
  use IngestionServiceWeb, :router

  pipeline :api do
    plug(:accepts, ["json"])
  end

  scope "/api", IngestionServiceWeb do
    pipe_through(:api)

    # Pipeline status and management
    get("/status", StatusController, :index)
    get("/metrics", MetricsController, :index)

    # Pipeline operations
    resources "/pipelines", PipelineController, except: [:new, :edit] do
      post("/start", PipelineController, :start)
      post("/stop", PipelineController, :stop)
      post("/restart", PipelineController, :restart)
    end

    # Metadata catalog endpoints
    scope "/metadata" do
      get("/", MetadataController, :index)
      post("/", MetadataController, :create)
      get("/search", MetadataController, :search)

      get("/:id", MetadataController, :show)
      put("/:id", MetadataController, :update)
      delete("/:id", MetadataController, :delete)

      post("/:id/tags", MetadataController, :add_tags)
      delete("/:id/tags", MetadataController, :remove_tags)

      get("/:id/lineage", MetadataController, :lineage)
      post("/lineage", MetadataController, :record_lineage)
    end
  end

  # Enable LiveDashboard in development
  if Application.compile_env(:ingestion_service, :dev_routes) do
    # If you want to use the LiveDashboard in production, you should put
    # it behind authentication and allow only admins to access it.
    # If your application does not have an admins-only section yet,
    # you can use Plug.BasicAuth to set up some basic authentication
    # as long as you are also using SSL (which you should anyway).
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through([:fetch_session, :protect_from_forgery])
      live_dashboard("/dashboard", metrics: IngestionServiceWeb.Telemetry)
    end
  end
end
