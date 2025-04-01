defmodule WebServiceWeb.Components.FileWatcher.StatisticsComponent do
  use WebServiceWeb, :live_component
  import WebServiceWeb.Components.FileHelpers

  @doc """
  Renders a statistics component for the file watcher sidebar.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.StatisticsComponent}
        id="statistics"
        current_folder={@current_folder}
        stats={@stats}
      />
  """
  def render(assigns) do
    ~H"""
    <div class="mt-6">
      <h4 class="text-md font-semibold mb-3 flex items-center text-gray-700">
        <i class="fas fa-chart-pie mr-2 text-blue-500"></i>
        Statistics
      </h4>
      <div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
        <div class="text-sm space-y-3">
          <div class="flex justify-between items-center">
            <span class="text-gray-600">Files</span>
            <span class="font-semibold text-gray-800"><%= @stats[@current_folder].file_count %></span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-gray-600">Directories</span>
            <span class="font-semibold text-gray-800"><%= @stats[@current_folder].dir_count %></span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-gray-600">Total Size</span>
            <span class="font-semibold text-gray-800"><%= format_file_size(@stats[@current_folder].total_size) %></span>
          </div>
          
          <!-- File type distribution -->
          <div class="mt-4 pt-3 border-t border-gray-200">
            <p class="text-xs font-semibold mb-2 text-gray-700">File Types</p>
            
            <%= for {type, count} <- @stats[@current_folder].type_counts do %>
              <div class="flex items-center mb-2">
                <div class={"w-3 h-3 rounded-full mr-2 #{file_type_color(type)}"}></div>
                <span class="text-xs flex-1"><%= type %></span>
                <span class="text-xs font-semibold"><%= count %></span>
              </div>
            <% end %>
          </div>
        </div>
      </div>
    </div>
    """
  end
end
