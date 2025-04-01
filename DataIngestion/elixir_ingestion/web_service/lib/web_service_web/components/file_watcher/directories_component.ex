defmodule WebServiceWeb.Components.FileWatcher.DirectoriesComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders directories in either grid or list view for the file watcher.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.DirectoriesComponent}
        id="directories"
        current_dirs={@current_dirs}
        current_path={@current_path}
        view_mode={@view_mode}
      />
  """
  def render(assigns) do
    ~H"""
    <div class="mb-8">
      <h3 class="text-lg font-semibold mb-3 flex items-center text-gray-800">
        <i class="fas fa-folder mr-2 text-yellow-500"></i>
        Directories
        <span class="ml-2 text-sm font-normal text-gray-500">(<%= length(@current_dirs) %>)</span>
      </h3>
      
      <%= if @view_mode == "grid" do %>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          <%= for dir <- @current_dirs do %>
            <button
              phx-click="navigate-path"
              phx-value-path={if @current_path == "", do: dir.name, else: "#{@current_path}/#{dir.name}"}
              class="flex flex-col h-32 items-center justify-center p-4 rounded-lg bg-white border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all">
              <div class="bg-yellow-50 p-3 rounded-full mb-2">
                <i class="fas fa-folder text-yellow-500 text-xl"></i>
              </div>
              <p class="font-medium text-center truncate w-full"><%= dir.name %></p>
              <p class="text-xs text-gray-500 mt-1">
                <%= NaiveDateTime.to_string(dir.last_modified) %>
              </p>
            </button>
          <% end %>
        </div>
      <% else %>
        <div class="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <ul class="divide-y divide-gray-200">
            <%= for dir <- @current_dirs do %>
              <li>
                <button
                  phx-click="navigate-path"
                  phx-value-path={if @current_path == "", do: dir.name, else: "#{@current_path}/#{dir.name}"}
                  class="w-full flex items-center px-4 py-3 hover:bg-gray-50 transition-colors">
                  <i class="fas fa-folder text-yellow-500 mr-3 text-lg"></i>
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-gray-900 truncate"><%= dir.name %></p>
                    <p class="text-xs text-gray-500">
                      <%= NaiveDateTime.to_string(dir.last_modified) %>
                    </p>
                  </div>
                  <i class="fas fa-chevron-right text-gray-400"></i>
                </button>
              </li>
            <% end %>
          </ul>
        </div>
      <% end %>
    </div>
    """
  end
end
