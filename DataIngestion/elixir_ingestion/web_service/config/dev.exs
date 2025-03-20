import Config

# Configure the endpoint for development
config :web_service, WebServiceWeb.Endpoint,
  http: [ip: {0, 0, 0, 0}, port: 4000],
  check_origin: false,
  code_reloader: true,
  debug_errors: true,
  secret_key_base: "8iHysm4eBDxe/PbVcEDmwm3O3fPPnxG1MaDnGnLUPX3kOyvgRxGyjQHlO8Nx7xtf",
  watchers: [
    esbuild: {Esbuild, :install_and_run, [:default, ~w(--sourcemap=inline --watch)]},
    tailwind: {Tailwind, :install_and_run, [:default, ~w(--watch)]}
  ]
  # Temporarily disabled to fix server startup issues
  # live_reload: [
  #   patterns: [
  #     ~r"priv/static/.*(js|css|png|jpeg|jpg|gif|svg)$",
  #     ~r"lib/web_service_web/(controllers|live|components)/.*(ex|heex)$"
  #   ]
  # ]

# Do not include metadata nor timestamps in development logs
config :logger, :console, format: "[$level] $message\n"

# Initialize plugs at runtime for faster development compilation
config :phoenix, :plug_init_mode, :runtime
