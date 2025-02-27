defmodule IngestionService.Pipeline.MetadataEnricher do
  @moduledoc """
  Pipeline stage for enriching and registering metadata.

  This module is responsible for extracting metadata from processed data,
  enriching it with additional information, and registering datasets with
  the metadata catalog, including tracking data lineage.
  """

  use GenStage
  require Logger
  alias IngestionService.Metadata.CatalogService

  @doc """
  Starts the metadata enricher stage.

  ## Parameters

  * `opts` - Options to pass to the stage
    * `:subscribe_to` - Stages to subscribe to
    * `:name` - The name of the stage

  ## Returns

  * `{:ok, pid}` - The PID of the started process
  * `{:error, reason}` - If there was an error starting the process
  """
  def start_link(opts) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenStage.start_link(__MODULE__, opts, name: name)
  end

  @impl true
  def init(opts) do
    Logger.info("Starting MetadataEnricher")
    subscribe_to = Keyword.get(opts, :subscribe_to, [])

    state = %{
      processed_count: 0
    }

    {:consumer_producer, state, subscribe_to: subscribe_to}
  end

  @impl true
  def handle_events(events, _from, state) do
    start_time = System.monotonic_time()

    {enriched_events, state} =
      Enum.map_reduce(events, state, fn event, acc_state ->
        {enriched_event, updated_state} = process_event(event, acc_state)
        {enriched_event, updated_state}
      end)

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :pipeline, :metadata_enricher],
      %{
        duration: duration,
        count: length(events)
      },
      %{
        stage: :metadata_enricher
      }
    )

    {:noreply, enriched_events, state}
  end

  # Private functions

  defp process_event(event, state) do
    start_time = System.monotonic_time()

    # Extract metadata from the event
    metadata = extract_metadata(event)

    # Register or update dataset in catalog
    register_dataset_result = register_or_update_dataset(metadata, event)

    # Track lineage if source datasets are available
    _ = track_lineage(register_dataset_result, event)

    # Increment processed count
    state = Map.update!(state, :processed_count, &(&1 + 1))

    # Add the dataset_id to the event
    event =
      case register_dataset_result do
        {:ok, dataset} ->
          Map.put(event, :dataset_id, dataset.id)

        _ ->
          event
      end

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :pipeline, :metadata_enricher, :process_event],
      %{duration: duration},
      %{
        event_id: Map.get(event, :id, "unknown"),
        status: elem(register_dataset_result, 0)
      }
    )

    {event, state}
  end

  defp extract_metadata(event) do
    # Get basic file information
    file_info = get_file_info(event)

    # Extract schema information
    schema = Map.get(event, :schema, %{})

    # Get quality metrics if available
    quality_metrics = Map.get(event, :quality_metrics, %{})

    # Construct dataset metadata
    %{
      name: Map.get(event, :filename),
      source: Map.get(event, :source, "file"),
      description:
        Map.get(event, :description, "Data ingested from #{Map.get(event, :filename)}"),
      format: get_format(event),
      row_count: Map.get(event, :row_count, 0),
      size_bytes: file_info.size || 0,
      schema: schema,
      quality_metrics: quality_metrics,
      metadata: %{
        ingested_at: DateTime.utc_now(),
        ingestion_id: Map.get(event, :id),
        file_info: file_info
      }
    }
  end

  defp get_file_info(event) do
    file_path = Map.get(event, :file_path)

    if file_path && File.exists?(file_path) do
      case File.stat(file_path) do
        {:ok, stat} ->
          %{
            size: stat.size,
            modified: stat.mtime,
            accessed: stat.atime,
            created: stat.ctime,
            path: file_path
          }

        {:error, reason} ->
          Logger.warning("Could not get file stats for #{file_path}: #{reason}")
          %{path: file_path}
      end
    else
      %{}
    end
  end

  defp get_format(event) do
    cond do
      Map.get(event, :format) ->
        Map.get(event, :format)

      file_path = Map.get(event, :file_path) ->
        Path.extname(file_path)
        |> String.downcase()
        |> String.replace_prefix(".", "")

      true ->
        "unknown"
    end
  end

  defp register_or_update_dataset(metadata, event) do
    # Check if a dataset with this name already exists
    case CatalogService.get_dataset(metadata.name) do
      {:ok, existing_dataset} ->
        # Update existing dataset
        CatalogService.update_dataset(existing_dataset.id, metadata)

      {:error, :not_found} ->
        # Register new dataset
        CatalogService.register_dataset(metadata)
    end
  end

  defp track_lineage({:ok, dataset}, event) do
    # Check if this dataset was derived from other datasets
    source_datasets = Map.get(event, :source_datasets, [])

    # Track lineage for each source dataset
    Enum.each(source_datasets, fn source_dataset ->
      lineage_attrs = %{
        operation: Map.get(source_dataset, :operation, "derived"),
        transformation_details: Map.get(source_dataset, :transformation_details, %{}),
        confidence: Map.get(source_dataset, :confidence, 1.0)
      }

      case CatalogService.get_dataset(source_dataset.name) do
        {:ok, parent_dataset} ->
          CatalogService.record_lineage(
            parent_dataset.id,
            dataset.id,
            lineage_attrs
          )

        {:error, _} ->
          Logger.warning(
            "Could not find source dataset #{source_dataset.name} for lineage tracking"
          )
      end
    end)
  end

  defp track_lineage(_, _), do: :ok
end
