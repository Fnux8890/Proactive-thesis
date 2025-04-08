defmodule Pipeline.MixProject do
  use Mix.Project

  def project do
    [
      app: :pipeline,
      version: "0.1.0",
      elixir: "~> 1.18.3",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      releases: [
        pipeline: [
          include_executables_for: [:unix],
          applications: [runtime_tools: :permanent]
        ]
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Pipeline.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # Client dependencies
      {:jason, "~> 1.2"},
      {:redix, "~> 1.1"},
      {:poolboy, "~> 1.5"},

      # Processing dependencies
      {:gen_stage, "~> 1.1"},
      {:nimble_csv, "~> 1.2"},

      # File handling dependencies
      {:file_system, "~> 0.2"},
      {:uuid, "~> 1.1"},

      # Logging dependencies
      {:logger_file_backend, "~> 0.0.13"},

      # Testing dependencies
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.30", only: :dev, runtime: false},
      {:mox, "~> 1.0", only: :test},
      {:mock, "~> 0.3.0", only: :test},
      {:meck, "~> 0.9.2", only: :test}
    ]
  end
end
