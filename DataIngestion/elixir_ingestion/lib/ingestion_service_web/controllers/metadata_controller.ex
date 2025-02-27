defmodule IngestionServiceWeb.MetadataController do
  @moduledoc """
  Controller for metadata catalog API endpoints.

  This module defines the API endpoints for interacting with the metadata catalog,
  including dataset registration, updating, searching, and lineage tracking.
  """

  use IngestionServiceWeb, :controller
  alias IngestionService.Metadata.CatalogService

  action_fallback(IngestionServiceWeb.FallbackController)

  @doc """
  Lists all datasets with optional filtering.

  ## Parameters

  * `conn` - The connection struct
  * `params` - Query parameters for filtering

  ## Returns

  * JSON response with datasets
  """
  def index(conn, params) do
    filters = extract_filters(params)

    with {:ok, datasets} <- CatalogService.list_datasets(filters) do
      render(conn, :index, datasets: datasets)
    end
  end

  @doc """
  Gets a dataset by its ID or name.

  ## Parameters

  * `conn` - The connection struct
  * `%{"id" => id}` - Map containing the dataset ID or name

  ## Returns

  * JSON response with the dataset
  """
  def show(conn, %{"id" => id}) do
    with {:ok, dataset} <- CatalogService.get_dataset(id) do
      render(conn, :show, dataset: dataset)
    end
  end

  @doc """
  Creates a new dataset.

  ## Parameters

  * `conn` - The connection struct
  * `%{"dataset" => dataset_params}` - Map containing the dataset parameters

  ## Returns

  * JSON response with the created dataset
  """
  def create(conn, %{"dataset" => dataset_params}) do
    with {:ok, dataset} <- CatalogService.register_dataset(dataset_params) do
      conn
      |> put_status(:created)
      |> put_resp_header("location", ~p"/api/metadata/#{dataset.id}")
      |> render(:show, dataset: dataset)
    end
  end

  @doc """
  Updates a dataset.

  ## Parameters

  * `conn` - The connection struct
  * `%{"id" => id, "dataset" => dataset_params}` - Map containing the dataset ID and parameters

  ## Returns

  * JSON response with the updated dataset
  """
  def update(conn, %{"id" => id, "dataset" => dataset_params}) do
    with {:ok, dataset} <- CatalogService.update_dataset(id, dataset_params) do
      render(conn, :show, dataset: dataset)
    end
  end

  @doc """
  Deletes a dataset.

  ## Parameters

  * `conn` - The connection struct
  * `%{"id" => id}` - Map containing the dataset ID

  ## Returns

  * JSON response with the deleted dataset
  """
  def delete(conn, %{"id" => id}) do
    with {:ok, dataset} <- CatalogService.delete_dataset(id) do
      render(conn, :show, dataset: dataset)
    end
  end

  @doc """
  Searches for datasets.

  ## Parameters

  * `conn` - The connection struct
  * `%{"q" => query}` - Map containing the search query

  ## Returns

  * JSON response with the search results
  """
  def search(conn, %{"q" => query}) do
    with {:ok, datasets} <- CatalogService.search_datasets(query) do
      render(conn, :index, datasets: datasets)
    end
  end

  @doc """
  Adds tags to a dataset.

  ## Parameters

  * `conn` - The connection struct
  * `%{"id" => id, "tags" => tags}` - Map containing the dataset ID and tags

  ## Returns

  * JSON response with the updated dataset
  """
  def add_tags(conn, %{"id" => id, "tags" => tags}) do
    with {:ok, dataset} <- CatalogService.add_tags(id, tags) do
      render(conn, :show, dataset: dataset)
    end
  end

  @doc """
  Removes tags from a dataset.

  ## Parameters

  * `conn` - The connection struct
  * `%{"id" => id, "tags" => tags}` - Map containing the dataset ID and tags

  ## Returns

  * JSON response with the updated dataset
  """
  def remove_tags(conn, %{"id" => id, "tags" => tags}) do
    with {:ok, dataset} <- CatalogService.remove_tags(id, tags) do
      render(conn, :show, dataset: dataset)
    end
  end

  @doc """
  Gets the lineage graph for a dataset.

  ## Parameters

  * `conn` - The connection struct
  * `%{"id" => id}` - Map containing the dataset ID
  * `params` - Additional query parameters for lineage options

  ## Returns

  * JSON response with the lineage graph
  """
  def lineage(conn, %{"id" => id} = params) do
    options = [
      direction: get_direction(params),
      depth: get_depth(params)
    ]

    with {:ok, graph} <- CatalogService.get_lineage(id, options) do
      render(conn, :lineage, lineage: graph)
    end
  end

  @doc """
  Records a lineage relationship between datasets.

  ## Parameters

  * `conn` - The connection struct
  * `%{"parent_id" => parent_id, "child_id" => child_id, "lineage" => lineage_params}` - Map containing lineage information

  ## Returns

  * JSON response with the created lineage
  """
  def record_lineage(conn, %{
        "parent_id" => parent_id,
        "child_id" => child_id,
        "lineage" => lineage_params
      }) do
    with {:ok, lineage} <- CatalogService.record_lineage(parent_id, child_id, lineage_params) do
      conn
      |> put_status(:created)
      |> render(:lineage_record, lineage: lineage)
    end
  end

  # Private helper functions

  defp extract_filters(params) do
    filters = %{}

    filters =
      if format = params["format"] do
        Map.put(filters, :format, format)
      else
        filters
      end

    filters =
      if created_after = params["created_after"] do
        case DateTime.from_iso8601(created_after) do
          {:ok, date, _} ->
            Map.put(filters, :created_after, date)

          _ ->
            filters
        end
      else
        filters
      end

    filters =
      if created_before = params["created_before"] do
        case DateTime.from_iso8601(created_before) do
          {:ok, date, _} ->
            Map.put(filters, :created_before, date)

          _ ->
            filters
        end
      else
        filters
      end

    filters =
      if tag = params["tag"] do
        Map.put(filters, :tag, tag)
      else
        filters
      end

    filters =
      if tags = params["tags"] do
        tags_list = String.split(tags, ",")
        Map.put(filters, :tags, tags_list)
      else
        filters
      end

    filters
  end

  defp get_direction(params) do
    case params["direction"] do
      "parents" -> :parents
      "children" -> :children
      _ -> :both
    end
  end

  defp get_depth(params) do
    case params["depth"] do
      nil ->
        3

      depth ->
        case Integer.parse(depth) do
          {value, _} -> max(1, min(value, 10))
          :error -> 3
        end
    end
  end
end
