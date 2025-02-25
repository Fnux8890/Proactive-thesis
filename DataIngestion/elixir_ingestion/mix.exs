defmodule IngestionService.MixProject do
  use Mix.Project

  def project do
    [
      app: :ingestion_service,
      version: "0.1.0",
      elixir: "~> 1.16",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps()
    ]
  end

  # Configuration for the OTP application
  def application do
    [
      mod: {IngestionService.Application, []},
      extra_applications: [:logger, :runtime_tools, :os_mon]
    ]
  end

  # Specifies which paths to compile per environment
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies
  defp deps do
    [
      # Phoenix framework for API
      {:phoenix, "~> 1.7.10"},
      {:phoenix_live_dashboard, "~> 0.8.3"},
      {:esbuild, "~> 0.8", runtime: Mix.env() == :dev},
      {:tailwind, "~> 0.2", runtime: Mix.env() == :dev},
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},
      {:jason, "~> 1.4"},
      {:plug_cowboy, "~> 2.6"},
      
      # GenStage for concurrent data processing
      {:gen_stage, "~> 1.2"},
      {:flow, "~> 1.2"},
      
      # Ecto for database interactions
      {:ecto_sql, "~> 3.10"},
      {:postgrex, "~> 0.17"},
      
      # Redis for caching and pub/sub
      {:redix, "~> 1.2"},
      {:castore, "~> 1.0"},
      
      # HTTP client for calling Python service
      {:finch, "~> 0.16"},
      
      # CSV/JSON/Excel parsing
      {:nimble_csv, "~> 1.2"},
      {:csv, "~> 3.2"},
      {:sweet_xml, "~> 0.7"},
      {:xlsxir, "~> 1.6"},
      
      # File system monitoring
      {:file_system, "~> 1.0"},
      
      # Monitoring and telemetry
      {:telemetry, "~> 1.2"},
      
      # Development tools
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project
  defp aliases do
    [
      setup: ["deps.get", "ecto.setup"],
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"],
      "assets.setup": ["tailwind.install --if-missing", "esbuild.install --if-missing"],
      "assets.build": ["tailwind default", "esbuild default"],
      "assets.deploy": ["tailwind default --minify", "esbuild default --minify", "phx.digest"]
    ]
  end
end 