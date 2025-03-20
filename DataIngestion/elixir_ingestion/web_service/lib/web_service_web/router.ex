defmodule WebServiceWeb.Router do
  use Phoenix.Router, helpers: false
  import Plug.Conn
  import Phoenix.Controller
  import Phoenix.LiveView.Router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {WebServiceWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", WebServiceWeb do
    pipe_through :browser

    live "/", DashboardLive, :index
    live "/jobs/:id", JobDetailLive, :show
    
    # File Watcher routes
    live "/file_watcher", FileWatcherLive
    live "/file_watcher/:folder", FileWatcherLive
    live "/file_watcher/:folder/:path", FileWatcherLive
    live "/file_watcher/:folder/:path/:file", FileWatcherLive
    
    # Pipeline management routes
    get "/pipeline", PipelineController, :index
    post "/pipeline/process", PipelineController, :process
    get "/pipeline/results/:file", PipelineController, :view_result
    get "/health", PipelineController, :health
  end

  # Other scopes may use custom stacks.
  scope "/api", WebServiceWeb do
    pipe_through :api
    
    post "/process", ApiController, :process_file
    get "/jobs", ApiController, :list_jobs
    get "/jobs/:id", ApiController, :get_job
  end

  # Enable LiveDashboard in development
  if Mix.env() == :dev do
    import Phoenix.LiveDashboard.Router

    scope "/" do
      pipe_through :browser
      live_dashboard "/dashboard", metrics: WebServiceWeb.Telemetry
    end
  end
end
