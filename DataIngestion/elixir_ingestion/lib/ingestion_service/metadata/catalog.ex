defmodule IngestionService.Metadata.Catalog do
  @moduledoc """
  Provides a central registry for dataset metadata.

  This module is responsible for:
  - Registering datasets in the catalog with comprehensive metadata
  - Updating dataset metadata with new information
  - Retrieving and querying dataset metadata
  - Providing data lineage tracking
  - Supporting tagging and annotations

  The catalog integrates with the existing pipeline components and enables
  data discovery, governance, and auditing capabilities.
  """

  use GenServer
  require Logger
  alias IngestionService.Repo
  alias IngestionService.Metadata.Dataset

  # Client API

  @doc """
  Starts the metadata catalog service.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Registers a new dataset in the catalog with the provided metadata.

  ## Parameters

  * `name` - The unique name of the dataset
  * `metadata` - A map containing dataset metadata including:
    * `:source` - Source of the data (e.g., file path, API, etc.)
    * `:schema` - Schema information (column names, types, etc.)
    * `:tags` - List of tags for categorization
    * `:description` - Human-readable description
    * `:created_at` - Timestamp of dataset creation
    * `:quality_metrics` - Data quality information
    * `:format` - Data format (CSV, JSON, etc.)
    * `:row_count` - Number of rows in the dataset
    * `:size_bytes` - Size of the dataset in bytes

  ## Returns

  * `{:ok, dataset_id}` - If the dataset was successfully registered
  * `{:error, reason}` - If registration failed
  """
  def register_dataset(name, metadata) do
    GenServer.call(__MODULE__, {:register_dataset, name, metadata})
  end

  @doc """
  Updates metadata for an existing dataset.

  ## Parameters

  * `id_or_name` - The ID or name of the dataset to update
  * `metadata` - A map containing updated metadata fields

  ## Returns

  * `{:ok, dataset}` - If the dataset was successfully updated
  * `{:error, reason}` - If update failed
  """
  def update_dataset(id_or_name, metadata) do
    GenServer.call(__MODULE__, {:update_dataset, id_or_name, metadata})
  end

  @doc """
  Gets a dataset by its ID or name.

  ## Parameters

  * `id_or_name` - The ID or name of the dataset

  ## Returns

  * `{:ok, dataset}` - If the dataset was found
  * `{:error, :not_found}` - If no dataset with the given ID or name exists
  """
  def get_dataset(id_or_name) do
    GenServer.call(__MODULE__, {:get_dataset, id_or_name})
  end

  @doc """
  Lists all datasets in the catalog, with optional filtering.

  ## Parameters

  * `filters` - A map containing filter criteria (e.g., %{format: "csv"})

  ## Returns

  * `[datasets]` - List of datasets matching the criteria
  """
  def list_datasets(filters \\ %{}) do
    GenServer.call(__MODULE__, {:list_datasets, filters})
  end

  @doc """
  Searches for datasets using a query string.

  ## Parameters

  * `query` - The search query string

  ## Returns

  * `[datasets]` - List of datasets matching the search query
  """
  def search_datasets(query) do
    GenServer.call(__MODULE__, {:search_datasets, query})
  end

  @doc """
  Adds tags to a dataset.

  ## Parameters

  * `id_or_name` - The ID or name of the dataset
  * `tags` - List of tags to add

  ## Returns

  * `{:ok, dataset}` - If tags were successfully added
  * `{:error, reason}` - If tag addition failed
  """
  def add_tags(id_or_name, tags) do
    GenServer.call(__MODULE__, {:add_tags, id_or_name, tags})
  end

  @doc """
  Removes tags from a dataset.

  ## Parameters

  * `id_or_name` - The ID or name of the dataset
  * `tags` - List of tags to remove

  ## Returns

  * `{:ok, dataset}` - If tags were successfully removed
  * `{:error, reason}` - If tag removal failed
  """
  def remove_tags(id_or_name, tags) do
    GenServer.call(__MODULE__, {:remove_tags, id_or_name, tags})
  end

  @doc """
  Records a lineage relationship between datasets.

  ## Parameters

  * `child_id_or_name` - The ID or name of the derived dataset
  * `parent_id_or_name` - The ID or name of the source dataset
  * `transformation` - Description of the transformation applied

  ## Returns

  * `{:ok, lineage_id}` - If lineage was successfully recorded
  * `{:error, reason}` - If lineage recording failed
  """
  def record_lineage(child_id_or_name, parent_id_or_name, transformation) do
    GenServer.call(
      __MODULE__,
      {:record_lineage, child_id_or_name, parent_id_or_name, transformation}
    )
  end

  @doc """
  Gets the lineage graph for a dataset (both parents and children).

  ## Parameters

  * `id_or_name` - The ID or name of the dataset
  * `depth` - How many levels to traverse (default: 1)

  ## Returns

  * `{:ok, %{parents: [...], children: [...]}}` - The lineage graph
  * `{:error, reason}` - If retrieving lineage failed
  """
  def get_lineage(id_or_name, depth \\ 1) do
    GenServer.call(__MODULE__, {:get_lineage, id_or_name, depth})
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    Logger.info("Starting Metadata Catalog service")

    # Check if the database tables exist and create them if they don't
    ensure_database_setup()

    # Initialize telemetry
    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :start],
      %{system_time: System.system_time()},
      %{}
    )

    {:ok, %{}}
  end

  @impl true
  def handle_call({:register_dataset, name, metadata}, _from, state) do
    start_time = System.monotonic_time()

    # Merge provided metadata with default values
    metadata =
      Map.merge(
        %{
          created_at: DateTime.utc_now(),
          updated_at: DateTime.utc_now(),
          tags: []
        },
        metadata
      )

    # Create dataset struct
    dataset_params = Map.merge(%{name: name}, metadata)

    result = create_dataset(dataset_params)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :register],
      %{duration: duration},
      %{dataset_name: name}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:update_dataset, id_or_name, metadata}, _from, state) do
    start_time = System.monotonic_time()

    # Set updated_at timestamp
    metadata = Map.put(metadata, :updated_at, DateTime.utc_now())

    result = update_dataset_impl(id_or_name, metadata)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :update],
      %{duration: duration},
      %{dataset_id: id_or_name}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_dataset, id_or_name}, _from, state) do
    start_time = System.monotonic_time()

    result = get_dataset_impl(id_or_name)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :get],
      %{duration: duration},
      %{dataset_id: id_or_name}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:list_datasets, filters}, _from, state) do
    start_time = System.monotonic_time()

    result = list_datasets_impl(filters)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :list],
      %{duration: duration, count: length(result)},
      %{filters: inspect(filters)}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:search_datasets, query}, _from, state) do
    start_time = System.monotonic_time()

    result = search_datasets_impl(query)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :search],
      %{duration: duration, count: length(result)},
      %{query: query}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:add_tags, id_or_name, tags}, _from, state) do
    start_time = System.monotonic_time()

    result = add_tags_impl(id_or_name, tags)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :add_tags],
      %{duration: duration},
      %{dataset_id: id_or_name, tags: tags}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:remove_tags, id_or_name, tags}, _from, state) do
    start_time = System.monotonic_time()

    result = remove_tags_impl(id_or_name, tags)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :remove_tags],
      %{duration: duration},
      %{dataset_id: id_or_name, tags: tags}
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call(
        {:record_lineage, child_id_or_name, parent_id_or_name, transformation},
        _from,
        state
      ) do
    start_time = System.monotonic_time()

    result = record_lineage_impl(child_id_or_name, parent_id_or_name, transformation)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :record_lineage],
      %{duration: duration},
      %{
        child: child_id_or_name,
        parent: parent_id_or_name,
        transformation: transformation
      }
    )

    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_lineage, id_or_name, depth}, _from, state) do
    start_time = System.monotonic_time()

    result = get_lineage_impl(id_or_name, depth)

    # Track execution time
    duration = System.monotonic_time() - start_time

    :telemetry.execute(
      [:ingestion_service, :metadata, :catalog, :get_lineage],
      %{duration: duration},
      %{dataset_id: id_or_name, depth: depth}
    )

    {:reply, result, state}
  end

  # Private implementation functions

  defp ensure_database_setup do
    # This function ensures that the database tables required by the catalog exist
    # In a production application, you would use migrations for this
    # For simplicity, we're checking and creating tables directly

    Logger.debug("Ensuring database tables for metadata catalog")

    # Normally these would be proper migrations
    # This is just a simplified example
    Repo.query!(
      "
      CREATE TABLE IF NOT EXISTS metadata_datasets (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        source TEXT,
        description TEXT,
        format TEXT,
        row_count INTEGER,
        size_bytes BIGINT,
        schema JSONB,
        tags TEXT[],
        quality_metrics JSONB,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
        metadata JSONB
      )
    ",
      []
    )

    Repo.query!(
      "
      CREATE TABLE IF NOT EXISTS metadata_lineage (
        id SERIAL PRIMARY KEY,
        child_id INTEGER REFERENCES metadata_datasets(id),
        parent_id INTEGER REFERENCES metadata_datasets(id),
        transformation TEXT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
        UNIQUE(child_id, parent_id)
      )
    ",
      []
    )

    # Create indexes for performance
    Repo.query!(
      "
      CREATE INDEX IF NOT EXISTS idx_metadata_datasets_name ON metadata_datasets (name);
      CREATE INDEX IF NOT EXISTS idx_metadata_datasets_format ON metadata_datasets (format);
      CREATE INDEX IF NOT EXISTS idx_metadata_datasets_tags ON metadata_datasets USING GIN (tags);
      CREATE INDEX IF NOT EXISTS idx_metadata_lineage_child ON metadata_lineage (child_id);
      CREATE INDEX IF NOT EXISTS idx_metadata_lineage_parent ON metadata_lineage (parent_id);
    ",
      []
    )

    # Create text search capabilities
    Repo.query!(
      "
      CREATE INDEX IF NOT EXISTS idx_metadata_datasets_text_search
      ON metadata_datasets
      USING GIN (to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, '')));
    ",
      []
    )
  end

  defp create_dataset(params) do
    # Extract known fields
    {known_fields, extra_metadata} = extract_known_fields(params)

    # Add any extra fields to the metadata JSONB field
    params_with_metadata = Map.put(known_fields, :metadata, extra_metadata)

    # Insert into database
    result =
      Repo.query(
        "INSERT INTO metadata_datasets
       (name, source, description, format, row_count, size_bytes, schema,
        tags, quality_metrics, created_at, updated_at, metadata)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
       RETURNING id",
        [
          params_with_metadata.name,
          params_with_metadata.source,
          params_with_metadata.description,
          params_with_metadata.format,
          params_with_metadata.row_count,
          params_with_metadata.size_bytes,
          Jason.encode!(params_with_metadata.schema || %{}),
          params_with_metadata.tags || [],
          Jason.encode!(params_with_metadata.quality_metrics || %{}),
          params_with_metadata.created_at,
          params_with_metadata.updated_at,
          Jason.encode!(params_with_metadata.metadata || %{})
        ]
      )

    case result do
      {:ok, %{rows: [[id]]}} ->
        {:ok, id}

      {:error, %{postgres: %{code: :unique_violation}}} ->
        {:error, :already_exists}

      {:error, error} ->
        Logger.error("Failed to create dataset: #{inspect(error)}")
        {:error, :database_error}
    end
  end

  defp update_dataset_impl(id_or_name, params) do
    # First, find the dataset
    case find_dataset_id(id_or_name) do
      {:ok, id} ->
        # Extract known fields
        {known_fields, extra_metadata} = extract_known_fields(params)

        # Build SET clause and params for the SQL update
        {set_clauses, values, _index} =
          known_fields
          |> Enum.reduce({[], [], 1}, fn {key, value}, {clauses, values, index} ->
            if value != nil do
              # For schema and quality_metrics, encode as JSON
              value =
                case key do
                  :schema -> Jason.encode!(value)
                  :quality_metrics -> Jason.encode!(value)
                  _ -> value
                end

              {["#{key} = $#{index}" | clauses], [value | values], index + 1}
            else
              {clauses, values, index}
            end
          end)

        # If there are extra metadata fields, update the metadata field as JSONB
        {set_clauses, values, index} =
          if map_size(extra_metadata) > 0 do
            # Get current metadata and merge
            {:ok, %{rows: [[current_metadata_json]]}} =
              Repo.query("SELECT metadata FROM metadata_datasets WHERE id = $1", [id])

            current_metadata =
              case Jason.decode(current_metadata_json) do
                {:ok, metadata} -> metadata
                _ -> %{}
              end

            # Merge with new metadata
            merged_metadata = Map.merge(current_metadata, extra_metadata)

            {["metadata = $#{index}" | set_clauses], [Jason.encode!(merged_metadata) | values],
             index + 1}
          else
            {set_clauses, values, index}
          end

        # Execute the update
        if length(set_clauses) > 0 do
          set_sql = Enum.join(set_clauses, ", ")

          result =
            Repo.query(
              "UPDATE metadata_datasets SET #{set_sql} WHERE id = $#{index} RETURNING *",
              values ++ [id]
            )

          case result do
            {:ok, %{rows: [row]}} ->
              {:ok, dataset_row_to_map(row)}

            {:error, error} ->
              Logger.error("Failed to update dataset: #{inspect(error)}")
              {:error, :database_error}
          end
        else
          # No fields to update
          get_dataset_impl(id)
        end

      error ->
        error
    end
  end

  defp get_dataset_impl(id_or_name) do
    # Find by ID or name
    query =
      cond do
        is_integer(id_or_name) ->
          {"SELECT * FROM metadata_datasets WHERE id = $1", [id_or_name]}

        is_binary(id_or_name) ->
          {"SELECT * FROM metadata_datasets WHERE name = $1", [id_or_name]}

        true ->
          raise ArgumentError, "id_or_name must be an integer or string"
      end

    {sql, params} = query

    case Repo.query(sql, params) do
      {:ok, %{rows: [row]}} ->
        {:ok, dataset_row_to_map(row)}

      {:ok, %{rows: []}} ->
        {:error, :not_found}

      {:error, error} ->
        Logger.error("Failed to get dataset: #{inspect(error)}")
        {:error, :database_error}
    end
  end

  defp list_datasets_impl(filters) do
    # Build WHERE clause from filters
    {where_clauses, values} =
      filters
      |> Enum.reduce({[], []}, fn
        {:format, format}, {clauses, values} ->
          {["format = $#{length(values) + 1}" | clauses], [format | values]}

        {:created_after, date}, {clauses, values} ->
          {["created_at > $#{length(values) + 1}" | clauses], [date | values]}

        {:created_before, date}, {clauses, values} ->
          {["created_at < $#{length(values) + 1}" | clauses], [date | values]}

        {:tag, tag}, {clauses, values} ->
          {["$#{length(values) + 1} = ANY(tags)" | clauses], [tag | values]}

        {:tags, tags}, {clauses, values} when is_list(tags) ->
          {["tags && $#{length(values) + 1}" | clauses], [tags | values]}

        _, acc ->
          # Ignore unknown filters
          acc
      end)

    # Construct the SQL query
    where_sql =
      if length(where_clauses) > 0, do: "WHERE #{Enum.join(where_clauses, " AND ")}", else: ""

    case Repo.query(
           "SELECT * FROM metadata_datasets #{where_sql} ORDER BY created_at DESC",
           values
         ) do
      {:ok, %{rows: rows}} ->
        Enum.map(rows, &dataset_row_to_map/1)

      {:error, error} ->
        Logger.error("Failed to list datasets: #{inspect(error)}")
        []
    end
  end

  defp search_datasets_impl(query) do
    # Use PostgreSQL full-text search
    search_query = "%#{query}%"

    case Repo.query(
           "SELECT * FROM metadata_datasets
       WHERE name ILIKE $1 OR description ILIKE $1
       ORDER BY created_at DESC",
           [search_query]
         ) do
      {:ok, %{rows: rows}} ->
        Enum.map(rows, &dataset_row_to_map/1)

      {:error, error} ->
        Logger.error("Failed to search datasets: #{inspect(error)}")
        []
    end
  end

  defp add_tags_impl(id_or_name, tags) do
    # First, find the dataset
    case find_dataset_id(id_or_name) do
      {:ok, id} ->
        # Get current tags
        {:ok, %{rows: [[current_tags]]}} =
          Repo.query("SELECT tags FROM metadata_datasets WHERE id = $1", [id])

        # Merge with new tags and remove duplicates
        merged_tags = Enum.uniq(current_tags ++ tags)

        # Update tags
        result =
          Repo.query(
            "UPDATE metadata_datasets SET tags = $1, updated_at = $2 WHERE id = $3 RETURNING *",
            [merged_tags, DateTime.utc_now(), id]
          )

        case result do
          {:ok, %{rows: [row]}} ->
            {:ok, dataset_row_to_map(row)}

          {:error, error} ->
            Logger.error("Failed to add tags: #{inspect(error)}")
            {:error, :database_error}
        end

      error ->
        error
    end
  end

  defp remove_tags_impl(id_or_name, tags) do
    # First, find the dataset
    case find_dataset_id(id_or_name) do
      {:ok, id} ->
        # Get current tags
        {:ok, %{rows: [[current_tags]]}} =
          Repo.query("SELECT tags FROM metadata_datasets WHERE id = $1", [id])

        # Remove specified tags
        updated_tags = current_tags -- tags

        # Update tags
        result =
          Repo.query(
            "UPDATE metadata_datasets SET tags = $1, updated_at = $2 WHERE id = $3 RETURNING *",
            [updated_tags, DateTime.utc_now(), id]
          )

        case result do
          {:ok, %{rows: [row]}} ->
            {:ok, dataset_row_to_map(row)}

          {:error, error} ->
            Logger.error("Failed to remove tags: #{inspect(error)}")
            {:error, :database_error}
        end

      error ->
        error
    end
  end

  defp record_lineage_impl(child_id_or_name, parent_id_or_name, transformation) do
    # Find both dataset IDs
    with {:ok, child_id} <- find_dataset_id(child_id_or_name),
         {:ok, parent_id} <- find_dataset_id(parent_id_or_name) do
      # Record the lineage relationship
      result =
        Repo.query(
          "INSERT INTO metadata_lineage (child_id, parent_id, transformation, created_at)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (child_id, parent_id)
         DO UPDATE SET transformation = $3, created_at = $4
         RETURNING id",
          [child_id, parent_id, transformation, DateTime.utc_now()]
        )

      case result do
        {:ok, %{rows: [[id]]}} ->
          {:ok, id}

        {:error, error} ->
          Logger.error("Failed to record lineage: #{inspect(error)}")
          {:error, :database_error}
      end
    else
      {:error, :not_found} ->
        {:error, :dataset_not_found}

      error ->
        error
    end
  end

  defp get_lineage_impl(id_or_name, depth) do
    # Limit depth to prevent overly large queries
    depth = min(depth, 5)

    # Find the dataset ID
    case find_dataset_id(id_or_name) do
      {:ok, id} ->
        # Get parents recursively
        parents = get_lineage_parents(id, depth)

        # Get children recursively
        children = get_lineage_children(id, depth)

        {:ok, %{parents: parents, children: children}}

      error ->
        error
    end
  end

  # Helper functions

  defp find_dataset_id(id) when is_integer(id) do
    # Check if the dataset exists
    case Repo.query("SELECT 1 FROM metadata_datasets WHERE id = $1", [id]) do
      {:ok, %{rows: [_]}} -> {:ok, id}
      {:ok, %{rows: []}} -> {:error, :not_found}
      {:error, _} -> {:error, :database_error}
    end
  end

  defp find_dataset_id(name) when is_binary(name) do
    # Find the ID by name
    case Repo.query("SELECT id FROM metadata_datasets WHERE name = $1", [name]) do
      {:ok, %{rows: [[id]]}} -> {:ok, id}
      {:ok, %{rows: []}} -> {:error, :not_found}
      {:error, _} -> {:error, :database_error}
    end
  end

  defp extract_known_fields(params) do
    # Define the known fields
    known_fields = [
      :name,
      :source,
      :description,
      :format,
      :row_count,
      :size_bytes,
      :schema,
      :tags,
      :quality_metrics,
      :created_at,
      :updated_at
    ]

    # Split params into known fields and extra metadata
    Enum.reduce(params, {%{}, %{}}, fn {key, value}, {known, extra} ->
      if Enum.member?(known_fields, key) do
        {Map.put(known, key, value), extra}
      else
        {known, Map.put(extra, key, value)}
      end
    end)
  end

  defp dataset_row_to_map(row) do
    # Convert DB row to map
    # The indices depend on the order of columns in the SELECT statement
    %{
      id: Enum.at(row, 0),
      name: Enum.at(row, 1),
      source: Enum.at(row, 2),
      description: Enum.at(row, 3),
      format: Enum.at(row, 4),
      row_count: Enum.at(row, 5),
      size_bytes: Enum.at(row, 6),
      schema: decode_jsonb(Enum.at(row, 7)),
      tags: Enum.at(row, 8) || [],
      quality_metrics: decode_jsonb(Enum.at(row, 9)),
      created_at: Enum.at(row, 10),
      updated_at: Enum.at(row, 11),
      metadata: decode_jsonb(Enum.at(row, 12))
    }
  end

  defp decode_jsonb(nil), do: %{}

  defp decode_jsonb(json) do
    case Jason.decode(json) do
      {:ok, data} -> data
      _ -> %{}
    end
  end

  defp get_lineage_parents(dataset_id, depth, current_depth \\ 0, visited \\ MapSet.new()) do
    # Prevent cycles and respect depth limit
    if current_depth >= depth || MapSet.member?(visited, dataset_id) do
      []
    else
      visited = MapSet.put(visited, dataset_id)

      # Get immediate parents
      case Repo.query(
             "SELECT p.*, l.transformation
         FROM metadata_lineage l
         JOIN metadata_datasets p ON l.parent_id = p.id
         WHERE l.child_id = $1",
             [dataset_id]
           ) do
        {:ok, %{rows: rows}} ->
          Enum.map(rows, fn row ->
            # Extract the transformation which is the last column
            transformation = List.last(row)
            # The rest is the dataset
            dataset = dataset_row_to_map(Enum.drop(row, -1))
            parent_id = dataset.id

            # Recursively get parents of parents
            parents = get_lineage_parents(parent_id, depth, current_depth + 1, visited)

            # Return node with its parents
            Map.put(dataset, :transformation, transformation)
            |> Map.put(:parents, parents)
          end)

        {:error, _} ->
          []
      end
    end
  end

  defp get_lineage_children(dataset_id, depth, current_depth \\ 0, visited \\ MapSet.new()) do
    # Prevent cycles and respect depth limit
    if current_depth >= depth || MapSet.member?(visited, dataset_id) do
      []
    else
      visited = MapSet.put(visited, dataset_id)

      # Get immediate children
      case Repo.query(
             "SELECT c.*, l.transformation
         FROM metadata_lineage l
         JOIN metadata_datasets c ON l.child_id = c.id
         WHERE l.parent_id = $1",
             [dataset_id]
           ) do
        {:ok, %{rows: rows}} ->
          Enum.map(rows, fn row ->
            # Extract the transformation which is the last column
            transformation = List.last(row)
            # The rest is the dataset
            dataset = dataset_row_to_map(Enum.drop(row, -1))
            child_id = dataset.id

            # Recursively get children of children
            children = get_lineage_children(child_id, depth, current_depth + 1, visited)

            # Return node with its children
            Map.put(dataset, :transformation, transformation)
            |> Map.put(:children, children)
          end)

        {:error, _} ->
          []
      end
    end
  end
end
