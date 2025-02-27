defmodule IngestionService.Metadata.Lineage do
  @moduledoc """
  Schema for dataset lineage in the metadata catalog.

  This module defines the Ecto schema for lineage relationships between datasets,
  tracking how datasets are derived from one another and the operations applied.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query
  alias IngestionService.Metadata.Dataset
  alias IngestionService.Metadata.Lineage

  @type t :: %__MODULE__{}

  @primary_key {:id, :id, autogenerate: true}
  schema "metadata_lineage" do
    field(:operation, :string)
    field(:transformation_details, :map, default: %{})
    field(:confidence, :float, default: 1.0)

    # Relationship fields
    belongs_to(:parent, Dataset)
    belongs_to(:child, Dataset)

    timestamps(type: :utc_datetime)
  end

  @required_fields ~w(parent_id child_id operation)a
  @optional_fields ~w(transformation_details confidence)a

  @doc """
  Creates a changeset for a lineage relationship.

  ## Parameters

  * `lineage` - The lineage to create a changeset for, or a new lineage if nil
  * `attrs` - The attributes to set on the lineage

  ## Returns

  * A changeset for the lineage
  """
  def changeset(lineage \\ %Lineage{}, attrs) do
    lineage
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_number(:confidence, greater_than: 0, less_than_or_equal_to: 1)
    |> foreign_key_constraint(:parent_id)
    |> foreign_key_constraint(:child_id)
    |> unique_constraint([:parent_id, :child_id, :operation])
  end

  @doc """
  Gets a lineage relationship by its ID.

  ## Parameters

  * `id` - The ID of the lineage to get

  ## Returns

  * The lineage if found
  * nil if not found
  """
  def get(id) do
    IngestionService.Repo.get(Lineage, id)
  end

  @doc """
  Lists all lineage relationships for a dataset, either as parent or child.

  ## Parameters

  * `dataset_id` - The ID of the dataset
  * `direction` - Either :parents, :children, or :both (default)

  ## Returns

  * A list of lineage relationships for the dataset
  """
  def list_for_dataset(dataset_id, direction \\ :both) do
    case direction do
      :parents ->
        from(l in Lineage,
          where: l.child_id == ^dataset_id,
          preload: [:parent]
        )
        |> IngestionService.Repo.all()

      :children ->
        from(l in Lineage,
          where: l.parent_id == ^dataset_id,
          preload: [:child]
        )
        |> IngestionService.Repo.all()

      :both ->
        parents = list_for_dataset(dataset_id, :parents)
        children = list_for_dataset(dataset_id, :children)
        parents ++ children
    end
  end

  @doc """
  Lists all lineage relationships between a parent and child dataset.

  ## Parameters

  * `parent_id` - The ID of the parent dataset
  * `child_id` - The ID of the child dataset

  ## Returns

  * A list of lineage relationships between the datasets
  """
  def list_between(parent_id, child_id) do
    from(l in Lineage,
      where: l.parent_id == ^parent_id and l.child_id == ^child_id
    )
    |> IngestionService.Repo.all()
  end

  @doc """
  Creates a new lineage relationship.

  ## Parameters

  * `attrs` - The attributes of the lineage to create

  ## Returns

  * `{:ok, lineage}` - If the lineage was created successfully
  * `{:error, changeset}` - If there was an error creating the lineage
  """
  def create(attrs) do
    %Lineage{}
    |> changeset(attrs)
    |> IngestionService.Repo.insert()
  end

  @doc """
  Updates a lineage relationship.

  ## Parameters

  * `lineage` - The lineage to update
  * `attrs` - The attributes to update

  ## Returns

  * `{:ok, lineage}` - If the lineage was updated successfully
  * `{:error, changeset}` - If there was an error updating the lineage
  """
  def update(lineage, attrs) do
    lineage
    |> changeset(attrs)
    |> IngestionService.Repo.update()
  end

  @doc """
  Deletes a lineage relationship.

  ## Parameters

  * `lineage` - The lineage to delete

  ## Returns

  * `{:ok, lineage}` - If the lineage was deleted successfully
  * `{:error, changeset}` - If there was an error deleting the lineage
  """
  def delete(lineage) do
    IngestionService.Repo.delete(lineage)
  end

  @doc """
  Builds a lineage graph for a dataset.

  ## Parameters

  * `dataset_id` - The ID of the dataset
  * `direction` - Either :parents, :children, or :both (default)
  * `depth` - Maximum depth of the graph (default 3)

  ## Returns

  * A map representing the lineage graph
  """
  def build_graph(dataset_id, direction \\ :both, depth \\ 3) do
    build_graph_recursive(dataset_id, direction, depth, 0, %{})
  end

  # Private recursive function to build the lineage graph
  defp build_graph_recursive(_dataset_id, _direction, max_depth, current_depth, graph)
       when current_depth >= max_depth do
    graph
  end

  defp build_graph_recursive(dataset_id, direction, max_depth, current_depth, graph) do
    # Get the dataset
    dataset = Dataset.get(dataset_id)

    # Skip if dataset not found or already in graph
    if is_nil(dataset) or Map.has_key?(graph, dataset_id) do
      graph
    else
      # Add dataset to graph
      graph =
        Map.put(graph, dataset_id, %{
          id: dataset_id,
          name: dataset.name,
          parents: %{},
          children: %{}
        })

      # Get lineage relationships
      relationships = list_for_dataset(dataset_id, direction)

      # Group by direction
      {parent_relations, child_relations} =
        Enum.split_with(relationships, &(&1.child_id == dataset_id))

      # Process parent relationships if needed
      graph =
        if direction in [:parents, :both] do
          parent_relations
          |> Enum.reduce(graph, fn relation, acc_graph ->
            # Get parent dataset
            parent_id = relation.parent_id

            # Add connection to current dataset's parents
            acc_graph =
              update_in(acc_graph[dataset_id].parents, fn parents ->
                Map.put(parents, parent_id, %{
                  operation: relation.operation,
                  details: relation.transformation_details
                })
              end)

            # Recursively build graph for parent
            build_graph_recursive(parent_id, direction, max_depth, current_depth + 1, acc_graph)
          end)
        else
          graph
        end

      # Process child relationships if needed
      graph =
        if direction in [:children, :both] do
          child_relations
          |> Enum.reduce(graph, fn relation, acc_graph ->
            # Get child dataset
            child_id = relation.child_id

            # Add connection to current dataset's children
            acc_graph =
              update_in(acc_graph[dataset_id].children, fn children ->
                Map.put(children, child_id, %{
                  operation: relation.operation,
                  details: relation.transformation_details
                })
              end)

            # Recursively build graph for child
            build_graph_recursive(child_id, direction, max_depth, current_depth + 1, acc_graph)
          end)
        else
          graph
        end

      graph
    end
  end
end
