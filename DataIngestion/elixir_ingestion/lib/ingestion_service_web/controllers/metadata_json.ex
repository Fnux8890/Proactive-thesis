defmodule IngestionServiceWeb.MetadataJSON do
  @moduledoc """
  JSON views for metadata controller responses.

  This module defines rendering functions for the metadata controller's JSON responses,
  including dataset information, lineage graphs, and other metadata catalog operations.
  """

  alias IngestionService.Metadata.Dataset
  alias IngestionService.Metadata.Lineage

  @doc """
  Renders a list of datasets.

  ## Parameters

  * `%{datasets: datasets}` - Map containing the datasets to render

  ## Returns

  * Map with a list of rendered datasets
  """
  def index(%{datasets: datasets}) do
    %{data: Enum.map(datasets, &dataset_json/1)}
  end

  @doc """
  Renders a single dataset.

  ## Parameters

  * `%{dataset: dataset}` - Map containing the dataset to render

  ## Returns

  * Map with the rendered dataset
  """
  def show(%{dataset: dataset}) do
    %{data: dataset_json(dataset)}
  end

  @doc """
  Renders a lineage graph.

  ## Parameters

  * `%{lineage: lineage}` - Map containing the lineage graph to render

  ## Returns

  * Map with the rendered lineage graph
  """
  def lineage(%{lineage: lineage}) do
    %{data: lineage}
  end

  @doc """
  Renders a lineage record.

  ## Parameters

  * `%{lineage: lineage}` - Map containing the lineage record to render

  ## Returns

  * Map with the rendered lineage record
  """
  def lineage_record(%{lineage: lineage}) do
    %{data: lineage_json(lineage)}
  end

  # Private helper functions

  defp dataset_json(%Dataset{} = dataset) do
    %{
      id: dataset.id,
      name: dataset.name,
      source: dataset.source,
      description: dataset.description,
      format: dataset.format,
      row_count: dataset.row_count,
      size_bytes: dataset.size_bytes,
      schema: dataset.schema,
      tags: dataset.tags,
      quality_metrics: dataset.quality_metrics,
      metadata: dataset.metadata,
      inserted_at: dataset.inserted_at,
      updated_at: dataset.updated_at
    }
  end

  defp lineage_json(%Lineage{} = lineage) do
    %{
      id: lineage.id,
      parent_id: lineage.parent_id,
      child_id: lineage.child_id,
      operation: lineage.operation,
      transformation_details: lineage.transformation_details,
      confidence: lineage.confidence,
      inserted_at: lineage.inserted_at,
      updated_at: lineage.updated_at
    }
  end
end
