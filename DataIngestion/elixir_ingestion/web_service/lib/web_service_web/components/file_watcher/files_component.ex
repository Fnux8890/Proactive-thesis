defmodule WebServiceWeb.Components.FileWatcher.FilesComponent do
  use WebServiceWeb, :live_component
  import WebServiceWeb.Components.FileHelpers

  @doc """
  Renders files in either grid or list view for the file watcher.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.FilesComponent}
        id="files"
        current_files={@current_files}
        current_file={@current_file}
        view_mode={@view_mode}
      />
  """
  def render(assigns) do
    ~H"""
    <div>
      <h3 class="text-lg font-semibold mb-3 flex items-center text-gray-800">
        <i class="fas fa-file mr-2 text-blue-500"></i>
        Files
        <span class="ml-2 text-sm font-normal text-gray-500">(<%= length(@current_files) %>)</span>
      </h3>
      
      <%= if length(@current_files) == 0 do %>
        <div class="bg-white rounded-lg border border-dashed border-gray-300 py-10 px-6 text-center">
          <i class="fas fa-file-alt text-gray-400 text-3xl mb-3"></i>
          <p class="text-gray-500">No files in this directory</p>
          <p class="text-sm text-gray-400 mt-1">Upload files for processing by the pipeline</p>
        </div>
      <% else %>
        <%= if @view_mode == "grid" do %>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <%= for file <- @current_files do %>
              <button
                phx-click="select-file"
                phx-value-file={file.name}
                class={"flex flex-col h-36 p-4 rounded-lg border hover:shadow-md transition-all #{if @current_file == file.name, do: "bg-blue-50 border-blue-300", else: "bg-white border-gray-200 hover:border-blue-300"}"}>
                <div class="flex items-center justify-center h-16 mb-2">
                  <i class={"#{file_icon_class(file.type)} text-3xl"}></i>
                </div>
                <p class="font-medium text-center truncate w-full text-sm"><%= file.name %></p>
                <div class="flex justify-between items-center mt-2 text-xs text-gray-500">
                  <span class={"px-2 py-1 rounded-full #{if file.type == :csv, do: "bg-green-100 text-green-800", else: "bg-blue-100 text-blue-800"}"}>
                    <%= file.type %>
                  </span>
                  <span><%= format_file_size(file.size) %></span>
                </div>
              </button>
            <% end %>
          </div>
        <% else %>
          <div class="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div class="overflow-x-auto">
              <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                  <tr>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Size</th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Modified</th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                  <%= for file <- @current_files do %>
                    <tr class={"hover:bg-gray-50 cursor-pointer transition-colors #{if @current_file == file.name, do: "bg-blue-50"}"} phx-click="select-file" phx-value-file={file.name}>
                      <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                          <i class={"#{file_icon_class(file.type)} mr-3 text-lg"}></i>
                          <div class="text-sm font-medium text-gray-900"><%= file.name %></div>
                        </div>
                      </td>
                      <td class="px-6 py-4 whitespace-nowrap">
                        <span class={"px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full #{if file.type == :csv, do: "bg-green-100 text-green-800", else: "bg-blue-100 text-blue-800"}"}>
                          <%= file.type %>
                        </span>
                      </td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <%= format_file_size(file.size) %>
                      </td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <%= NaiveDateTime.to_string(file.last_modified) %>
                      </td>
                      <td class="px-6 py-4 whitespace-nowrap">
                        <!-- File status - can be enhanced later to show processing status -->
                        <span class="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">
                          Ready
                        </span>
                      </td>
                    </tr>
                  <% end %>
                </tbody>
              </table>
            </div>
          </div>
        <% end %>
      <% end %>
    </div>
    """
  end
end
