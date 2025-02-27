defmodule IngestionService.Metadata.CatalogService do
  @moduledoc """
  Service module for metadata catalog operations.

  This module provides a high-level API for interacting with the metadata catalog,
  including dataset registration, updating, searching, and lineage tracking.
  It serves as the main entry point for all metadata-related operations.
  """

  use GenServer
  require Logger
  alias IngestionService.Metadata.Dataset
  alias IngestionService.Metadata.Lineage

  # Client API

  @doc """
  Starts the metadata catalog service.

  ## Parameters

  * `opts` - Options to pass to the GenServer

  ## Returns

  * `{:ok, pid}` - The PID of the started process
  * `{:error, reason}` - If there was an error starting the process
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Registers a new dataset in the metadata catalog.

  ## Parameters

  * `dataset_attrs` - The attributes of the dataset to register

  ## Returns

  * `{:ok, dataset}` - If the dataset was registered successfully
  * `{:error, reason}` - If there was an error registering the dataset
  """
  def register_dataset(dataset_attrs) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:register_dataset, dataset_attrs})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :register_dataset],
      %{duration: duration},
      %{status: elem(result, 0)}
    )

    result
  end

  @doc """
  Updates an existing dataset in the metadata catalog.

  ## Parameters

  * `dataset_id` - The ID of the dataset to update
  * `dataset_attrs` - The attributes to update

  ## Returns

  * `{:ok, dataset}` - If the dataset was updated successfully
  * `{:error, reason}` - If there was an error updating the dataset
  """
  def update_dataset(dataset_id, dataset_attrs) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:update_dataset, dataset_id, dataset_attrs})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :update_dataset],
      %{duration: duration},
      %{status: elem(result, 0), dataset_id: dataset_id}
    )

    result
  end

  @doc """
  Gets a dataset by its ID or name.

  ## Parameters

  * `id_or_name` - The ID or name of the dataset to get

  ## Returns

  * `{:ok, dataset}` - If the dataset was found
  * `{:error, :not_found}` - If the dataset was not found
  """
  def get_dataset(id_or_name) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:get_dataset, id_or_name})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :get_dataset],
      %{duration: duration},
      %{status: elem(result, 0), id_or_name: id_or_name}
    )

    result
  end

  @doc """
  Lists all datasets with optional filtering.

  ## Parameters

  * `filters` - A map containing filter criteria

  ## Returns

  * `{:ok, datasets}` - A list of datasets matching the criteria
  """
  def list_datasets(filters \\ %{}) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:list_datasets, filters})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :list_datasets],
      %{duration: duration},
      %{status: elem(result, 0), filter_count: map_size(filters)}
    )

    result
  end

  @doc """
  Searches for datasets using a query string.

  ## Parameters

  * `query` - The search query string

  ## Returns

  * `{:ok, datasets}` - A list of datasets matching the search query
  """
  def search_datasets(query) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:search_datasets, query})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :search_datasets],
      %{duration: duration},
      %{status: elem(result, 0), query: query}
    )

    result
  end

  @doc """
  Adds tags to a dataset.

  ## Parameters

  * `dataset_id` - The ID of the dataset to tag
  * `tags` - A list of tags to add

  ## Returns

  * `{:ok, dataset}` - If the tags were added successfully
  * `{:error, reason}` - If there was an error adding the tags
  """
  def add_tags(dataset_id, tags) when is_list(tags) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:add_tags, dataset_id, tags})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :add_tags],
      %{duration: duration},
      %{status: elem(result, 0), dataset_id: dataset_id, tag_count: length(tags)}
    )

    result
  end

  @doc """
  Removes tags from a dataset.

  ## Parameters

  * `dataset_id` - The ID of the dataset to untag
  * `tags` - A list of tags to remove

  ## Returns

  * `{:ok, dataset}` - If the tags were removed successfully
  * `{:error, reason}` - If there was an error removing the tags
  """
  def remove_tags(dataset_id, tags) when is_list(tags) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:remove_tags, dataset_id, tags})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :remove_tags],
      %{duration: duration},
      %{status: elem(result, 0), dataset_id: dataset_id, tag_count: length(tags)}
    )

    result
  end

  @doc """
  Records a lineage relationship between datasets.

  ## Parameters

  * `parent_id` - The ID of the parent dataset
  * `child_id` - The ID of the child dataset
  * `lineage_attrs` - The attributes of the lineage relationship

  ## Returns

  * `{:ok, lineage}` - If the lineage was recorded successfully
  * `{:error, reason}` - If there was an error recording the lineage
  """
  def record_lineage(parent_id, child_id, lineage_attrs) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:record_lineage, parent_id, child_id, lineage_attrs})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :record_lineage],
      %{duration: duration},
      %{status: elem(result, 0), parent_id: parent_id, child_id: child_id}
    )

    result
  end

  @doc """
  Gets the lineage graph for a dataset.

  ## Parameters

  * `dataset_id` - The ID of the dataset
  * `options` - Options for retrieving the lineage
    * `:direction` - Either `:parents`, `:children`, or `:both` (default)
    * `:depth` - Maximum depth of the graph (default 3)

  ## Returns

  * `{:ok, graph}` - The lineage graph for the dataset
  * `{:error, reason}` - If there was an error retrieving the lineage
  """
  def get_lineage(dataset_id, options \\ []) do
    direction = Keyword.get(options, :direction, :both)
    depth = Keyword.get(options, :depth, 3)

    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:get_lineage, dataset_id, direction, depth})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :get_lineage],
      %{duration: duration},
      %{status: elem(result, 0), dataset_id: dataset_id, direction: direction, depth: depth}
    )

    result
  end

  @doc """
  Deletes a dataset and all of its lineage relationships.

  ## Parameters

  * `dataset_id` - The ID of the dataset to delete

  ## Returns

  * `{:ok, dataset}` - If the dataset was deleted successfully
  * `{:error, reason}` - If there was an error deleting the dataset
  """
  def delete_dataset(dataset_id) do
    start_time = System.monotonic_time()
    result = GenServer.call(__MODULE__, {:delete_dataset, dataset_id})

    end_time = System.monotonic_time()
    duration = System.convert_time_unit(end_time - start_time, :native, :millisecond)

    :telemetry.execute(
      [:ingestion_service, :metadata, :delete_dataset],
      %{duration: duration},
      %{status: elem(result, 0), dataset_id: dataset_id}
    )

    result
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    Logger.info("Starting Metadata Catalog Service")
    {:ok, %{}}
  end

  @impl true
  def handle_call({:register_dataset, dataset_attrs}, _from, state) do
    case Dataset.create(dataset_attrs) do
      {:ok, dataset} ->
        Logger.info("Registered dataset: #{dataset.name}")
        {:reply, {:ok, dataset}, state}

      {:error, changeset} ->
        Logger.error("Failed to register dataset: #{inspect(changeset.errors)}")
        {:reply, {:error, changeset}, state}
    end
  end

  @impl true
  def handle_call({:update_dataset, dataset_id, dataset_attrs}, _from, state) do
    with {:ok, dataset} <- get_dataset_by_id(dataset_id),
         {:ok, updated_dataset} <- Dataset.update(dataset, dataset_attrs) do
      Logger.info("Updated dataset: #{updated_dataset.name}")
      {:reply, {:ok, updated_dataset}, state}
    else
      {:error, reason} ->
        Logger.error("Failed to update dataset: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:get_dataset, id_or_name}, _from, state) do
    result = get_dataset_by_id_or_name(id_or_name)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:list_datasets, filters}, _from, state) do
    datasets = Dataset.list(filters)
    {:reply, {:ok, datasets}, state}
  end

  @impl true
  def handle_call({:search_datasets, query}, _from, state) do
    datasets = Dataset.search(query)
    {:reply, {:ok, datasets}, state}
  end

  @impl true
  def handle_call({:add_tags, dataset_id, tags}, _from, state) do
    with {:ok, dataset} <- get_dataset_by_id(dataset_id),
         # Ensure tags are unique
         updated_tags = Enum.uniq(dataset.tags ++ tags),
         {:ok, updated_dataset} <- Dataset.update(dataset, %{tags: updated_tags}) do
      Logger.info("Added tags to dataset #{dataset.name}: #{inspect(tags)}")
      {:reply, {:ok, updated_dataset}, state}
    else
      {:error, reason} ->
        Logger.error("Failed to add tags: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:remove_tags, dataset_id, tags}, _from, state) do
    with {:ok, dataset} <- get_dataset_by_id(dataset_id),
         updated_tags = dataset.tags -- tags,
         {:ok, updated_dataset} <- Dataset.update(dataset, %{tags: updated_tags}) do
      Logger.info("Removed tags from dataset #{dataset.name}: #{inspect(tags)}")
      {:reply, {:ok, updated_dataset}, state}
    else
      {:error, reason} ->
        Logger.error("Failed to remove tags: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:record_lineage, parent_id, child_id, lineage_attrs}, _from, state) do
    with {:ok, _parent} <- get_dataset_by_id(parent_id),
         {:ok, _child} <- get_dataset_by_id(child_id),
         attrs = Map.merge(%{parent_id: parent_id, child_id: child_id}, lineage_attrs),
         {:ok, lineage} <- Lineage.create(attrs) do
      Logger.info("Recorded lineage from #{parent_id} to #{child_id}")
      {:reply, {:ok, lineage}, state}
    else
      {:error, reason} ->
        Logger.error("Failed to record lineage: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:get_lineage, dataset_id, direction, depth}, _from, state) do
    with {:ok, _dataset} <- get_dataset_by_id(dataset_id),
         graph <- Lineage.build_graph(dataset_id, direction, depth) do
      {:reply, {:ok, graph}, state}
    else
      {:error, reason} ->
        Logger.error("Failed to get lineage: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:delete_dataset, dataset_id}, _from, state) do
    with {:ok, dataset} <- get_dataset_by_id(dataset_id),
         {:ok, _} <- Dataset.delete(dataset) do
      Logger.info("Deleted dataset: #{dataset.name}")
      {:reply, {:ok, dataset}, state}
    else
      {:error, reason} ->
        Logger.error("Failed to delete dataset: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  # Private helper functions

  defp get_dataset_by_id(id) when is_integer(id) or is_binary(id) do
    case Dataset.get(id) do
      nil -> {:error, :not_found}
      dataset -> {:ok, dataset}
    end
  end

  defp get_dataset_by_id_or_name(id)
       when is_integer(id) or (is_binary(id) and byte_size(id) <= 36) do
    case Dataset.get(id) do
      nil -> get_dataset_by_name(id)
      dataset -> {:ok, dataset}
    end
  end

  defp get_dataset_by_id_or_name(name) when is_binary(name) do
    get_dataset_by_name(name)
  end

  defp get_dataset_by_name(name) when is_binary(name) do
    case Dataset.get_by_name(name) do
      nil -> {:error, :not_found}
      dataset -> {:ok, dataset}
    end
  end
end
