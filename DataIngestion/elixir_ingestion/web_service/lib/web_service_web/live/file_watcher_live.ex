defmodule WebServiceWeb.FileWatcherLive do
  use WebServiceWeb, :live_view
  require Logger
  import WebServiceWeb.Components.FileHelpers
  alias WebService.Cache.RedisCache

  @data_path "/app/data"
  @folders ["aarslev", "knudjepsen"]
  @cache_ttl 300 # 5 minutes cache TTL

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Schedule periodic refresh of file list
      :timer.send_interval(10000, self(), :refresh_files)
    end

    {:ok, socket |> assign(:loading, true) |> assign(:view_mode, "grid") |> init_file_data()}
  end

  @impl true
def handle_params(params, _url, socket) do
  IO.inspect(params, label: "PARAMS")

  current_folder = params["folder"] || List.first(@folders)
  path_param = params["path"] || ""

  # Check if path is actually a file (contains extension)
  {current_path, current_file} =
    if String.contains?(path_param, ".") do
      # If path has a file extension, it's a file
      # Set current_path to empty (root of folder) and file to the path
      {"", path_param}
    else
      # Normal directory path
      {path_param, params["file"]}
    end

  IO.inspect({current_folder, current_path, current_file}, label: "PARSED_PARAMS")

  socket =
    socket
    |> assign(:current_folder, current_folder)
    |> assign(:current_path, current_path)
    |> assign(:current_file, current_file)
    |> assign(:loading, true)

  # Fetch data asynchronously
  Process.send_after(self(), {:load_data, current_folder, current_path, current_file}, 100)

  {:noreply, socket}
end

  @impl true
  def handle_event("select-folder", %{"folder" => folder}, socket) do
    {:noreply, push_patch(socket, to: "/file_watcher/#{folder}", replace: true)}
  end

  @impl true
  def handle_event("navigate-path", %{"path" => path}, socket) do
    current_folder = socket.assigns.current_folder
    {:noreply, push_patch(socket, to: "/file_watcher/#{current_folder}/#{path}", replace: true)}
  end

  @impl true
  def handle_event("select-file", %{"file" => file}, socket) do
    current_folder = socket.assigns.current_folder
    current_path = socket.assigns.current_path

    # If file is empty string, it means we are closing the preview
    if file == "" do
      {:noreply, push_patch(socket, to: "/file_watcher/#{current_folder}/#{current_path}", replace: true)}
    else
      {:noreply, push_patch(socket, to: "/file_watcher/#{current_folder}/#{current_path}/#{file}", replace: true)}
    end
  end

  @impl true
  def handle_event("set-view-mode", %{"mode" => mode}, socket) when mode in ["grid", "list"] do
    {:noreply, assign(socket, :view_mode, mode)}
  end

  @impl true
  def handle_event("refresh-files", _params, socket) do
    # Invalidate relevant caches
    RedisCache.delete("folder_data")

    current_folder = socket.assigns.current_folder
    current_path = socket.assigns.current_path
    RedisCache.delete("#{current_folder}:#{current_path}")

    socket = socket
      |> assign(:loading, true)

    # Refresh the data
    Process.send_after(self(), {:load_data, current_folder, current_path, socket.assigns[:current_file]}, 100)

    {:noreply, socket}
  end

  @impl true
  def handle_event("close-preview", _params, socket) do
    current_folder = socket.assigns.current_folder
    current_path = socket.assigns.current_path
    {:noreply, push_patch(socket, to: "/file_watcher/#{current_folder}/#{current_path}", replace: true)}
  end

  @impl true
  def handle_info(:refresh_files, socket) do
    # Selectively invalidate certain caches that might have changed
    RedisCache.delete("folder_data")


    current_folder = socket.assigns.current_folder
    current_path = socket.assigns.current_path
    RedisCache.delete("#{current_folder}:#{current_path}")

    {:noreply, refresh_file_data(socket)}
  end

  @impl true
  def handle_info({:load_data, _folder, _path, file}, socket) do
    socket =
      socket
      |> assign(:current_file, file)
      |> refresh_file_data()
      |> assign(:loading, false)

    {:noreply, socket}
  end

  defp init_file_data(socket) do
    folder_data =
      Enum.map(@folders, fn folder ->
        {folder, %{files: [], directories: [], path: ""}}
      end)
      |> Enum.into(%{})

    current_folder = List.first(@folders)

    socket
    |> assign(:folders, @folders)
    |> assign(:folder_data, folder_data)
    |> assign(:current_folder, current_folder)
    |> assign(:current_path, "")
    |> assign(:current_file, nil)
    |> assign(:current_files, [])
    |> assign(:current_dirs, [])
    |> assign(:file_content, nil)
    |> assign(:breadcrumbs, [%{name: current_folder, path: ""}])
    |> assign(:stats, %{})
  end

  defp refresh_file_data(socket) do
    current_folder = socket.assigns[:current_folder] || List.first(@folders)
    current_path = socket.assigns[:current_path] || ""
    current_file = socket.assigns[:current_file]

    IO.inspect({current_folder, current_path, current_file}, label: "REFRESH_DATA")

    # Try to get folder data from cache first
    folder_data =
      case get_cached_folder_data() do
        {:ok, cached_data} ->
          cached_data
        _ ->
          # Cache miss, load from filesystem and cache the result
          data = load_folder_data_from_filesystem()
          cache_folder_data(data)
          data
      end

    # Get the current path contents
    current_dir_path = Path.join([@data_path, current_folder, current_path])
    cache_key = "#{current_folder}:#{current_path}"

    # Get current files and directories, try cache first
    {current_files, current_dirs} =
      case RedisCache.get(cache_key) do
        {:ok, {files, dirs}} ->
          {files, dirs}
        _ ->
          # Cache miss, load from filesystem
          files = list_files(current_dir_path)
          dirs = list_directories(current_dir_path)
          # Cache the result
          RedisCache.put(cache_key, {files, dirs}, @cache_ttl)
          {files, dirs}
      end

    # Build breadcrumbs for navigation
    breadcrumbs = build_breadcrumbs(current_folder, current_path)

    # If there's a file to display, load its content (potentially from cache)
    file_content =
      if current_file do
        file_path = Path.join(current_dir_path, current_file)
        get_file_content(file_path)
      else
        nil
      end

    socket
    |> assign(:folders, @folders)
    |> assign(:folder_data, folder_data)
    |> assign(:current_folder, current_folder)
    |> assign(:current_path, current_path)
    |> assign(:current_file, current_file)
    |> assign(:current_files, current_files)
    |> assign(:current_dirs, current_dirs)
    |> assign(:file_content, file_content)
    |> assign(:breadcrumbs, breadcrumbs)
    |> assign(:stats, get_folder_stats(folder_data))
  end

  # Cache helpers
  defp get_cached_folder_data do
    RedisCache.get("folder_data")
  end

  defp cache_folder_data(folder_data) do
    RedisCache.put("folder_data", folder_data, @cache_ttl)
  end

  defp load_folder_data_from_filesystem do
    Enum.map(@folders, fn folder ->
      path = Path.join(@data_path, folder)
      {folder, list_directory_contents(path, "")}
    end)
    |> Enum.into(%{})
  end

  defp get_file_content(file_path) do
    file_cache_key = "file:#{file_path}"

    case RedisCache.get(file_cache_key) do
      {:ok, content} ->
        content
      _ ->
        # Cache miss, read from file
        content = read_file_content(file_path)
        # Only cache if it's not an error message and not too large
        if is_binary(content) &&
           !String.starts_with?(content, "Error") &&
           !String.starts_with?(content, "File too large") &&
           !String.starts_with?(content, "Binary file") do
          RedisCache.put(file_cache_key, content, @cache_ttl)
        end
        content
    end
  end

  defp build_breadcrumbs(folder, path) do
    base = [%{name: folder, path: ""}]

    if path == "" do
      base
    else
      parts = String.split(path, "/", trim: true)

      Enum.reduce(parts, base, fn part, acc ->
        previous_path = if length(acc) > 1 do
          acc |> List.last() |> Map.get(:path)
        else
          ""
        end

        new_path = if previous_path == "" do
          part
        else
          "#{previous_path}/#{part}"
        end

        acc ++ [%{name: part, path: new_path}]
      end)
    end
  end

  defp list_directory_contents(path, current_path) do
    %{
      files: list_files(path),
      directories: list_directories(path) |> Enum.map(fn dir ->
        dir_path = Path.join(path, dir.name)
        rel_path = if current_path == "", do: dir.name, else: "#{current_path}/#{dir.name}"
        Map.put(dir, :contents, list_directory_contents(dir_path, rel_path))
      end),
      path: current_path
    }
  end

  defp list_files(path) do
    case File.ls(path) do
      {:ok, files} ->
        files
        |> Enum.reject(fn file -> File.dir?(Path.join(path, file)) end)
        |> Enum.map(fn file ->
          file_path = Path.join(path, file)
          %{
            name: file,
            size: get_file_size(file_path),
            type: get_file_type(file),
            last_modified: get_file_modified_date(file_path)
          }
        end)
        |> Enum.sort_by(
          fn %{last_modified: date} -> date end,
          fn d1, d2 -> NaiveDateTime.compare(d1, d2) == :gt end
        )

      {:error, reason} ->
        Logger.error("Failed to list files in #{path}: #{reason}")
        []
    end
  end

  defp list_directories(path) do
    case File.ls(path) do
      {:ok, entries} ->
        entries
        |> Enum.filter(fn entry -> File.dir?(Path.join(path, entry)) end)
        |> Enum.map(fn dir ->
          dir_path = Path.join(path, dir)
          %{
            name: dir,
            last_modified: get_file_modified_date(dir_path)
          }
        end)
        |> Enum.sort_by(fn %{name: name} -> name end)

      {:error, reason} ->
        Logger.error("Failed to list directories in #{path}: #{reason}")
        []
    end
  end

  defp get_file_size(path) do
    case File.stat(path) do
      {:ok, %{size: size}} -> size
      _ -> 0
    end
  end

  defp get_file_modified_date(path) do
    case File.stat(path) do
      {:ok, %{mtime: mtime}} ->
        {{year, month, day}, {hour, minute, second}} = mtime
        {:ok, datetime} = NaiveDateTime.new(year, month, day, hour, minute, second)
        datetime
      _ -> NaiveDateTime.utc_now()
    end
  end

  defp get_file_type(filename) do
    case Path.extname(filename) |> String.downcase() do
      ".csv" -> :csv
      ".json" -> :json
      ".xlsx" -> :excel
      ".xls" -> :excel
      ".txt" -> :text
      ".md" -> :text
      ".xml" -> :xml
      _ -> :unknown
    end
  end

  defp read_file_content(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        # For binary files or large files, truncate content
        if String.valid?(content) && byte_size(content) < 500_000 do
          content
        else
          if byte_size(content) > 500_000 do
            "File too large to display. Size: #{format_file_size(byte_size(content))}"
          else
            "Binary file not displayed"
          end
        end
      {:error, reason} ->
        Logger.error("Failed to read file #{file_path}: #{reason}")
        "Error reading file: #{reason}"
    end
  end

  defp format_content(content, :csv, _path) do
    rows =
      content
      |> String.split("\n", trim: true)
      |> Enum.map(fn line -> String.split(line, ",") end)

    headers = List.first(rows) || []
    data = if length(rows) > 1, do: Enum.slice(rows, 1..-1), else: []

    %{
      type: :csv,
      headers: headers,
      data: data,
      preview: Enum.slice(data, 0..9) # Show first 10 rows
    }
  end

  defp format_content(content, :json, _path) do
    case Jason.decode(content) do
      {:ok, json} ->
        %{
          type: :json,
          data: json
        }
      {:error, _} ->
        %{
          type: :text,
          data: content
        }
    end
  end

  defp format_content(content, :text, _path) do
    %{
      type: :text,
      data: content
    }
  end

  defp format_content(_content, _type, path) do
    %{
      type: :unknown,
      data: "Preview not available for this file type: #{Path.extname(path)}"
    }
  end

  defp get_folder_stats(folder_data) do
    Enum.map(folder_data, fn {folder, data} ->
      file_count = count_files_recursive(data)
      total_size = calculate_total_size_recursive(data)
      dir_count = count_directories_recursive(data)

      # Group by file type
      type_counts = count_file_types_recursive(data)

      {folder, %{
        file_count: file_count,
        dir_count: dir_count,
        total_size: total_size,
        type_counts: type_counts
      }}
    end)
    |> Enum.into(%{})
  end

  defp count_files_recursive(%{files: files, directories: dirs}) do
    file_count = length(files)
    dir_files = Enum.reduce(dirs, 0, fn dir, acc ->
      acc + count_files_recursive(dir.contents)
    end)

    file_count + dir_files
  end

  defp count_directories_recursive(%{directories: dirs}) do
    dir_count = length(dirs)
    nested_dirs = Enum.reduce(dirs, 0, fn dir, acc ->
      acc + count_directories_recursive(dir.contents)
    end)

    dir_count + nested_dirs
  end

  defp calculate_total_size_recursive(%{files: files, directories: dirs}) do
    file_size = Enum.reduce(files, 0, fn %{size: size}, acc -> acc + size end)
    dir_size = Enum.reduce(dirs, 0, fn dir, acc ->
      acc + calculate_total_size_recursive(dir.contents)
    end)

    file_size + dir_size
  end

  defp count_file_types_recursive(%{files: files, directories: dirs}) do
    file_types = files
      |> Enum.group_by(fn %{type: type} -> type end)
      |> Enum.map(fn {type, files} -> {type, length(files)} end)
      |> Enum.into(%{})

    dir_types = Enum.reduce(dirs, %{}, fn dir, acc ->
      dir_type_counts = count_file_types_recursive(dir.contents)
      Map.merge(acc, dir_type_counts, fn _k, v1, v2 -> v1 + v2 end)
    end)

    Map.merge(file_types, dir_types, fn _k, v1, v2 -> v1 + v2 end)
  end
end
