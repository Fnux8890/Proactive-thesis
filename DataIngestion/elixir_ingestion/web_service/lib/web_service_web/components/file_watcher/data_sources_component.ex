defmodule WebServiceWeb.Components.FileWatcher.DataSourcesComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders a data sources sidebar component for the file watcher.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.DataSourcesComponent}
        id="data_sources"
        folders={@folders}
        current_folder={@current_folder}
        stats={@stats}
      />
  """
  def render(assigns) do
    ~H"""
    <div class="p-4">
      <h3 class="text-lg font-bold mb-4 text-gray-800">Data Sources</h3>
      <ul class="space-y-1 mb-6">
        <%= for folder <- @folders do %>
          <li>
            <button 
              phx-click="select-folder" 
              phx-value-folder={folder} 
              class={"flex items-center w-full p-2 rounded-md transition-colors #{if @current_folder == folder, do: "bg-blue-50 text-blue-700 border-l-4 border-blue-500", else: "hover:bg-gray-100 text-gray-700"}"}>
              <i class="fas fa-database mr-2 text-sm"></i>
              <span class="font-medium"><%= folder %></span>
              <%= if @stats[folder] do %>
                <span class="ml-auto text-xs font-semibold rounded-full px-2 py-1 bg-gray-100 text-gray-600">
                  <%= @stats[folder].file_count %>
                </span>
              <% end %>
            </button>
          </li>
        <% end %>
      </ul>
    </div>
    """
  end
end
