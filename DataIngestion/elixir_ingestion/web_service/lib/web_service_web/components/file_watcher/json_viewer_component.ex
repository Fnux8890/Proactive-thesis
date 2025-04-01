defmodule WebServiceWeb.Components.FileWatcher.JsonViewerComponent do
  use WebServiceWeb, :live_component
  
  @doc """
  Renders a JSON file in an interactive tree view format.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.JsonViewerComponent}
        id="json_viewer"
        json_data={@json_data}
        file_name={@current_file}
      />
  """
  def mount(socket) do
    {:ok, 
     socket
     |> assign(:expanded_paths, MapSet.new())
     |> assign(:search_term, "")
     |> assign(:json_structure, nil)
     |> assign(:error, nil)
     |> assign(:view_mode, "tree")
     |> assign(:matching_paths, MapSet.new())}
  end

  def update(assigns, socket) do
    socket = assign(socket, assigns)
    
    # Parse JSON data if available
    socket = 
      if Map.has_key?(socket.assigns, :json_data) && is_binary(socket.assigns.json_data) do
        case Jason.decode(socket.assigns.json_data) do
          {:ok, parsed_json} ->
            socket
            |> assign(:json_structure, parsed_json)
            |> assign(:error, nil)
            
          {:error, reason} ->
            socket
            |> assign(:json_structure, nil)
            |> assign(:error, "Invalid JSON format: #{inspect(reason)}")
        end
      else
        socket
        |> assign(:json_structure, nil)
        |> assign(:error, "No JSON data available")
      end
    
    {:ok, socket}
  end

  def handle_event("toggle-node", %{"path" => path}, socket) do
    expanded_paths = 
      if MapSet.member?(socket.assigns.expanded_paths, path) do
        MapSet.delete(socket.assigns.expanded_paths, path)
      else
        MapSet.put(socket.assigns.expanded_paths, path)
      end
    
    {:noreply, assign(socket, :expanded_paths, expanded_paths)}
  end

  def handle_event("expand-all", _, socket) do
    # Collect all possible paths in the JSON structure
    all_paths = collect_all_paths(socket.assigns.json_structure)
    
    {:noreply, assign(socket, :expanded_paths, MapSet.new(all_paths))}
  end

  def handle_event("collapse-all", _, socket) do
    {:noreply, assign(socket, :expanded_paths, MapSet.new())}
  end

  def handle_event("search", %{"search" => %{"term" => term}}, socket) do
    socket =
      if term && term != "" do
        matching_paths = find_matching_paths(socket.assigns.json_structure, term)
        
        # Automatically expand parent paths of matches
        expanded_paths = expand_parent_paths(matching_paths)
        
        socket
        |> assign(:search_term, term)
        |> assign(:matching_paths, matching_paths)
        |> assign(:expanded_paths, expanded_paths)
      else
        socket
        |> assign(:search_term, "")
        |> assign(:matching_paths, MapSet.new())
      end
      
    {:noreply, socket}
  end

  def handle_event("switch-view", %{"mode" => mode}, socket) do
    {:noreply, assign(socket, :view_mode, mode)}
  end

  def render(assigns) do
    ~H"""
    <div class="bg-white rounded-lg border border-gray-200 overflow-hidden mt-4">
      <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center px-4 py-3 bg-gray-50 border-b border-gray-200">
        <h3 class="text-lg font-semibold text-gray-800 flex items-center mb-2 sm:mb-0">
          <i class="fas fa-code mr-2 text-yellow-600"></i>
          <%= @file_name %>
        </h3>
        
        <div class="flex flex-col sm:flex-row gap-2 items-center">
          <form phx-submit="search" phx-target={@myself} class="relative">
            <input 
              type="text" 
              name="search[term]" 
              value={@search_term}
              placeholder="Search..."
              class="py-1 pl-8 pr-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" 
            />
            <i class="fas fa-search absolute left-2.5 top-2 text-gray-400 text-sm"></i>
          </form>
          
          <div class="flex bg-gray-100 rounded-lg p-1">
            <button 
              phx-click="switch-view"
              phx-value-mode="tree" 
              phx-target={@myself}
              class={"px-3 py-1 rounded-md text-sm transition-all #{if @view_mode == "tree", do: "bg-white shadow-sm font-medium", else: "hover:bg-gray-200"}"}>
              <i class="fas fa-sitemap mr-1"></i>
              Tree
            </button>
            <button 
              phx-click="switch-view"
              phx-value-mode="raw" 
              phx-target={@myself}
              class={"px-3 py-1 rounded-md text-sm transition-all #{if @view_mode == "raw", do: "bg-white shadow-sm font-medium", else: "hover:bg-gray-200"}"}>
              <i class="fas fa-code mr-1"></i>
              Raw
            </button>
          </div>

          <div class="flex gap-1">
            <button 
              phx-click="expand-all" 
              phx-target={@myself}
              class="bg-blue-50 hover:bg-blue-100 text-blue-700 text-sm py-1 px-2 rounded-md"
              disabled={@view_mode != "tree"}
            >
              <i class="fas fa-plus-square mr-1"></i>
              Expand All
            </button>
            
            <button 
              phx-click="collapse-all" 
              phx-target={@myself}
              class="bg-blue-50 hover:bg-blue-100 text-blue-700 text-sm py-1 px-2 rounded-md"
              disabled={@view_mode != "tree"}
            >
              <i class="fas fa-minus-square mr-1"></i>
              Collapse All
            </button>
          </div>
        </div>
      </div>
      
      <!-- JSON Viewer -->
      <div class="overflow-x-auto">
        <%= if @error do %>
          <div class="p-8 text-center text-gray-500">
            <i class="fas fa-exclamation-triangle text-yellow-500 text-2xl mb-2"></i>
            <p><%= @error %></p>
          </div>
        <% else %>
          <%= if @view_mode == "tree" do %>
            <div class="p-4 font-mono text-sm">
              <%= render_json_tree(@json_structure, [], @expanded_paths, @matching_paths, @search_term, @myself) %>
            </div>
          <% else %>
            <div class="p-4 overflow-auto max-h-96">
              <pre class="font-mono text-sm whitespace-pre-wrap">
                <%= Jason.encode!(@json_structure, pretty: true) %>
              </pre>
            </div>
          <% end %>
        <% end %>
      </div>
    </div>
    """
  end
  
  # Helper functions for rendering JSON tree
  def render_json_tree(data, path, expanded_paths, matching_paths, search_term, myself) do
    path_str = Enum.join(path, ".")
    is_expanded = MapSet.member?(expanded_paths, path_str)
    is_match = search_term != "" && MapSet.member?(matching_paths, path_str)
    
    assigns = %{
      data: data,
      path: path,
      path_str: path_str,
      expanded_paths: expanded_paths,
      matching_paths: matching_paths,
      search_term: search_term,
      is_expanded: is_expanded,
      is_match: is_match,
      myself: myself
    }
    
    ~H"""
    <%= cond do %>
      <% is_map(data) -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-center">
            <%= if map_size(data) > 0 do %>
              <button
                phx-click="toggle-node"
                phx-value-path={@path_str}
                phx-target={@myself}
                class="w-5 h-5 mr-1 text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <i class={"fas #{if @is_expanded, do: "fa-caret-down", else: "fa-caret-right"}"}></i>
              </button>
            <% else %>
              <div class="w-5 h-5 mr-1"></div>
            <% end %>
            
            <span class="font-semibold mr-1 text-blue-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "Object" %>
            </span>
            <span class="text-gray-500">{</span>
            <%= if map_size(data) > 0 && !@is_expanded do %>
              <span class="text-gray-400 ml-1">…</span>
            <% end %>
            <span class="text-gray-500"><%= if map_size(data) == 0 || !@is_expanded, do: "}", else: "" %></span>
            
            <%= if map_size(data) > 0 do %>
              <span class="ml-2 text-xs text-gray-400"><%= map_size(data) %> <%= if map_size(data) == 1, do: "property", else: "properties" %></span>
            <% end %>
          </div>
          
          <%= if map_size(data) > 0 && @is_expanded do %>
            <div class="pl-6 border-l border-gray-200 ml-2 my-1">
              <%= for {key, value} <- Enum.sort_by(data, fn {k, _} -> k end) do %>
                <div class="mt-1">
                  <%= render_json_tree(value, @path ++ [key], @expanded_paths, @matching_paths, @search_term, @myself) %>
                </div>
              <% end %>
            </div>
            <div><span class="text-gray-500">}</span></div>
          <% end %>
        </div>
      
      <% is_list(data) -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-center">
            <%= if length(data) > 0 do %>
              <button
                phx-click="toggle-node"
                phx-value-path={@path_str}
                phx-target={@myself}
                class="w-5 h-5 mr-1 text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <i class={"fas #{if @is_expanded, do: "fa-caret-down", else: "fa-caret-right"}"}></i>
              </button>
            <% else %>
              <div class="w-5 h-5 mr-1"></div>
            <% end %>
            
            <span class="font-semibold mr-1 text-purple-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "Array" %>
            </span>
            <span class="text-gray-500">[</span>
            <%= if length(data) > 0 && !@is_expanded do %>
              <span class="text-gray-400 ml-1">…</span>
            <% end %>
            <span class="text-gray-500"><%= if length(data) == 0 || !@is_expanded, do: "]", else: "" %></span>
            
            <%= if length(data) > 0 do %>
              <span class="ml-2 text-xs text-gray-400"><%= length(data) %> <%= if length(data) == 1, do: "item", else: "items" %></span>
            <% end %>
          </div>
          
          <%= if length(data) > 0 && @is_expanded do %>
            <div class="pl-6 border-l border-gray-200 ml-2 my-1">
              <%= for {value, index} <- Enum.with_index(data) do %>
                <div class="mt-1">
                  <%= render_json_tree(value, @path ++ ["#{index}"], @expanded_paths, @matching_paths, @search_term, @myself) %>
                </div>
              <% end %>
            </div>
            <div><span class="text-gray-500">]</span></div>
          <% end %>
        </div>
      
      <% is_binary(data) -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-start">
            <div class="w-5 h-5 mr-1"></div>
            <span class="font-semibold mr-1 text-green-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "String" %>:
            </span>
            <span class="text-green-600 break-all">"<%= truncate_value(data) %>"</span>
          </div>
        </div>
      
      <% is_number(data) -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-start">
            <div class="w-5 h-5 mr-1"></div>
            <span class="font-semibold mr-1 text-amber-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "Number" %>:
            </span>
            <span class="text-amber-600"><%= data %></span>
          </div>
        </div>
      
      <% is_boolean(data) -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-start">
            <div class="w-5 h-5 mr-1"></div>
            <span class="font-semibold mr-1 text-red-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "Boolean" %>:
            </span>
            <span class="text-red-600"><%= data %></span>
          </div>
        </div>
      
      <% is_nil(data) -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-start">
            <div class="w-5 h-5 mr-1"></div>
            <span class="font-semibold mr-1 text-gray-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "Null" %>:
            </span>
            <span class="text-gray-500">null</span>
          </div>
        </div>
      
      <% true -> %>
        <div class={if @is_match, do: "bg-yellow-100 -mx-1 px-1 py-0.5 rounded", else: ""}>
          <div class="flex items-start">
            <div class="w-5 h-5 mr-1"></div>
            <span class="font-semibold mr-1 text-gray-700">
              <%= if length(@path) > 0, do: List.last(@path), else: "Unknown" %>:
            </span>
            <span class="text-gray-500"><%= inspect(data) %></span>
          </div>
        </div>
    <% end %>
    """
  end

  defp truncate_value(string) when is_binary(string) do
    max_length = 100
    
    if String.length(string) > max_length do
      String.slice(string, 0, max_length) <> "..."
    else
      string
    end
  end

  defp collect_all_paths(data, current_path \\ []) do
    path_str = Enum.join(current_path, ".")
    
    cond do
      is_map(data) ->
        [path_str | Enum.flat_map(data, fn {key, value} ->
          collect_all_paths(value, current_path ++ [key])
        end)]
        
      is_list(data) ->
        [path_str | Enum.flat_map(Enum.with_index(data), fn {value, index} ->
          collect_all_paths(value, current_path ++ ["#{index}"])
        end)]
        
      true ->
        [path_str]
    end
  end

  defp find_matching_paths(data, search_term, current_path \\ []) do
    path_str = Enum.join(current_path, ".")
    search_term = String.downcase(search_term)
    
    matches = 
      cond do
        is_map(data) ->
          # Check if any key matches
          key_matches = 
            data
            |> Enum.filter(fn {key, _} -> 
              String.downcase(to_string(key)) =~ search_term
            end)
            |> Enum.map(fn {key, _} -> 
              Enum.join(current_path ++ [key], ".")
            end)
            
          # Recursively check values
          value_matches = 
            Enum.flat_map(data, fn {key, value} ->
              MapSet.to_list(find_matching_paths(value, search_term, current_path ++ [key]))
            end)
            
          key_matches ++ value_matches
          
        is_list(data) ->
          Enum.flat_map(Enum.with_index(data), fn {value, index} ->
            MapSet.to_list(find_matching_paths(value, search_term, current_path ++ ["#{index}"]))
          end)
          
        is_binary(data) ->
          if String.downcase(data) =~ search_term, do: [path_str], else: []
          
        is_number(data) || is_boolean(data) ->
          if String.downcase(to_string(data)) =~ search_term, do: [path_str], else: []
          
        true ->
          []
      end
    
    MapSet.new(matches)
  end

  defp expand_parent_paths(matching_paths) do
    paths = MapSet.to_list(matching_paths)
    
    parent_paths = 
      Enum.flat_map(paths, fn path ->
        path
        |> String.split(".")
        |> Enum.reduce({[], []}, fn segment, {segments, paths} ->
          new_segments = segments ++ [segment]
          {new_segments, paths ++ [Enum.join(new_segments, ".")]}
        end)
        |> elem(1)
      end)
    
    MapSet.new(parent_paths)
  end
end
