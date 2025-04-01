defmodule WebServiceWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :web_service

  # The session will be stored in the cookie and signed,
  # this means its contents can be read but not tampered with.
  # Set :encryption_salt if you would also like to encrypt it.
  @session_options [
    store: :cookie,
    key: "_web_service_key",
    signing_salt: "7jlQmPaX",
    same_site: "Lax"
  ]

  socket "/live", Phoenix.LiveView.Socket, websocket: [connect_info: [session: @session_options]]

  # Serve at "/" the static files from "priv/static" directory.
  plug Plug.Static,
    at: "/",
    from: :web_service,
    gzip: false,
    only: WebServiceWeb.static_paths()

  # Code reloading can be explicitly enabled under the
  # :code_reloader configuration of your endpoint.
  if code_reloading? do
    # Temporarily disable LiveReload to fix server startup issues
    # socket "/phoenix/live_reload/socket", Phoenix.LiveReload.Socket
    # The line below is commented out until phoenix_live_reload is properly set up
    # plug Phoenix.LiveReload.Plug
  end

  plug Phoenix.LiveDashboard.RequestLogger,
    param_key: "request_logger",
    cookie_key: "request_logger"

  plug Plug.RequestId
  plug Plug.Telemetry, event_prefix: [:phoenix, :endpoint]

  plug Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()

  plug Plug.MethodOverride
  plug Plug.Head
  plug Plug.Session, @session_options
  plug WebServiceWeb.Router
end
