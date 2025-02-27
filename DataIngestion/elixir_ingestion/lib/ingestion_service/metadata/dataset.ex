defmodule IngestionService.Metadata.Dataset do
  @moduledoc """
  Schema for Dataset metadata in the catalog.

  This module defines the Ecto schema for datasets stored in the metadata catalog,
  including validations and associations.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query
  alias IngestionService.Metadata.Dataset
  alias IngestionService.Metadata.Lineage

  @type t :: %__MODULE__{}

  @primary_key {:id, :id, autogenerate: true}
  schema "metadata_datasets" do
    field(:name, :string)
    field(:source, :string)
    field(:description, :string)
    field(:format, :string)
    field(:row_count, :integer)
    field(:size_bytes, :integer)
    field(:schema, :map, default: %{})
    field(:tags, {:array, :string}, default: [])
    field(:quality_metrics, :map, default: %{})
    field(:metadata, :map, default: %{})

    # Add timestamps for created_at and updated_at
    timestamps(type: :utc_datetime)

    # Define relationships for lineage
    has_many(:child_lineages, Lineage, foreign_key: :parent_id)
    has_many(:parent_lineages, Lineage, foreign_key: :child_id)
    has_many(:children, through: [:child_lineages, :child])
    has_many(:parents, through: [:parent_lineages, :parent])
  end

  @required_fields ~w(name)a
  @optional_fields ~w(source description format row_count size_bytes schema tags quality_metrics metadata)a

  @doc """
  Creates a changeset for a dataset.

  ## Parameters

  * `dataset` - The dataset to create a changeset for, or a new dataset if nil
  * `attrs` - The attributes to set on the dataset

  ## Returns

  * A changeset for the dataset
  """
  def changeset(dataset \\ %Dataset{}, attrs) do
    dataset
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint(:name)
    |> validate_number(:row_count, greater_than_or_equal_to: 0)
    |> validate_number(:size_bytes, greater_than_or_equal_to: 0)
  end

  @doc """
  Gets a dataset by its ID.

  ## Parameters

  * `id` - The ID of the dataset to get

  ## Returns

  * The dataset if found
  * nil if not found
  """
  def get(id) do
    IngestionService.Repo.get(Dataset, id)
  end

  @doc """
  Gets a dataset by its name.

  ## Parameters

  * `name` - The name of the dataset to get

  ## Returns

  * The dataset if found
  * nil if not found
  """
  def get_by_name(name) do
    IngestionService.Repo.get_by(Dataset, name: name)
  end

  @doc """
  Lists all datasets with optional filtering.

  ## Parameters

  * `filters` - A map containing filter criteria

  ## Returns

  * A list of datasets matching the criteria
  """
  def list(filters \\ %{}) do
    query = from(d in Dataset)

    query =
      Enum.reduce(filters, query, fn
        {:format, format}, query ->
          from(q in query, where: q.format == ^format)

        {:created_after, date}, query ->
          from(q in query, where: q.inserted_at > ^date)

        {:created_before, date}, query ->
          from(q in query, where: q.inserted_at < ^date)

        {:tag, tag}, query ->
          from(q in query, where: ^tag in q.tags)

        {:tags, tags}, query when is_list(tags) ->
          from(q in query, where: fragment("? && ?", q.tags, ^tags))

        _, query ->
          query
      end)

    IngestionService.Repo.all(query)
  end

  @doc """
  Searches for datasets.

  ## Parameters

  * `query` - The search query string

  ## Returns

  * A list of datasets matching the search query
  """
  def search(query) do
    search_term = "%#{query}%"

    from(d in Dataset,
      where: ilike(d.name, ^search_term) or ilike(d.description, ^search_term)
    )
    |> IngestionService.Repo.all()
  end

  @doc """
  Creates a new dataset.

  ## Parameters

  * `attrs` - The attributes of the dataset to create

  ## Returns

  * `{:ok, dataset}` - If the dataset was created successfully
  * `{:error, changeset}` - If there was an error creating the dataset
  """
  def create(attrs) do
    %Dataset{}
    |> changeset(attrs)
    |> IngestionService.Repo.insert()
  end

  @doc """
  Updates a dataset.

  ## Parameters

  * `dataset` - The dataset to update
  * `attrs` - The attributes to update

  ## Returns

  * `{:ok, dataset}` - If the dataset was updated successfully
  * `{:error, changeset}` - If there was an error updating the dataset
  """
  def update(dataset, attrs) do
    dataset
    |> changeset(attrs)
    |> IngestionService.Repo.update()
  end

  @doc """
  Deletes a dataset.

  ## Parameters

  * `dataset` - The dataset to delete

  ## Returns

  * `{:ok, dataset}` - If the dataset was deleted successfully
  * `{:error, changeset}` - If there was an error deleting the dataset
  """
  def delete(dataset) do
    IngestionService.Repo.delete(dataset)
  end
end
