defmodule WebServiceWeb.Components.FileWatcher.FilePreviewComponent do
  use WebServiceWeb, :live_component
  import WebServiceWeb.Components.FileHelpers

  @doc """
  Renders a file preview component for the file watcher.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.FilePreviewComponent}
        id="file_preview"
        current_file={@current_file}
        current_files={@current_files}
        file_content={@file_content}
      />
  """
  def render(assigns) do
    ~H"""
    <div class="mt-8 border-t pt-6">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-lg font-bold flex items-center text-gray-800">
          <i class={"mr-2 text-lg #{file_icon_class(get_file_type(@current_file))}"}></i>
          Preview: <%= @current_file %>
        </h3>
        
        <div class="flex items-center space-x-2">
          <button class="p-2 rounded-md hover:bg-gray-100 text-gray-600 transition-colors" phx-click="close-preview">
            <i class="fas fa-times"></i>
          </button>
        </div>
      </div>
      
      <!-- File information card -->
      <div class="bg-white p-4 rounded-lg border border-gray-200 mb-4">
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div class="flex flex-col">
            <span class="text-xs text-gray-500 mb-1">File Type</span>
            <span class="text-sm font-medium">
              <%= if String.contains?(@current_file, ".") do %>
                <%= @current_file |> String.split(".") |> List.last() |> String.upcase() %>
              <% else %>
                Unknown
              <% end %>
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-xs text-gray-500 mb-1">Size</span>
            <span class="text-sm font-medium">
              <%= @current_files |> Enum.find(fn f -> f.name == @current_file end) |> (fn file -> format_file_size(file.size) end).() %>
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-xs text-gray-500 mb-1">Last Modified</span>
            <span class="text-sm font-medium">
              <%= @current_files |> Enum.find(fn f -> f.name == @current_file end) |> (fn file -> NaiveDateTime.to_string(file.last_modified) end).() %>
            </span>
          </div>
        </div>
      </div>
      
      <!-- File content preview -->
      <%= cond do %>
        <% is_csv_file?(@current_file) && is_valid_content?(@file_content) -> %>
          <.live_component
            module={WebServiceWeb.Components.FileWatcher.CsvViewerComponent}
            id="csv_viewer"
            csv_data={@file_content}
            file_name={@current_file}
          />
        
        <% is_json_file?(@current_file) && is_valid_content?(@file_content) -> %>
          <.live_component
            module={WebServiceWeb.Components.FileWatcher.JsonViewerComponent}
            id="json_viewer"
            json_data={@file_content}
            file_name={@current_file}
          />
        
        <% is_binary(@file_content) -> %>
          <%= if String.starts_with?(@file_content, "File too large") || String.starts_with?(@file_content, "Binary file") || String.starts_with?(@file_content, "Error") do %>
            <div class="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 rounded-lg flex items-start">
              <i class="fas fa-exclamation-triangle mt-1 mr-3"></i>
              <div>
                <p class="font-medium mb-1">Preview Unavailable</p>
                <p><%= @file_content %></p>
              </div>
            </div>
          <% else %>
            <!-- Default text content preview -->
            <div class="bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
              <div class="flex justify-between items-center px-4 py-2 bg-gray-100 border-b border-gray-200">
                <span class="text-sm font-medium text-gray-700">File Content</span>
              </div>
              <div class="overflow-auto max-h-96 p-4 font-mono text-sm whitespace-pre">
                <%= @file_content %>
              </div>
            </div>
          <% end %>
        
        <% true -> %>
          <div class="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg flex items-start">
            <i class="fas fa-times-circle mt-1 mr-3"></i>
            <div>
              <p class="font-medium mb-1">Error Loading File</p>
              <p>Unable to display file content. The file may be corrupted or in an unsupported format.</p>
            </div>
          </div>
      <% end %>
      
      <!-- Pipeline integration hint card -->
      <div class="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div class="flex items-start">
          <i class="fas fa-info-circle text-blue-500 mt-1 mr-3"></i>
          <div>
            <h4 class="font-medium text-blue-800 mb-1">Pipeline Integration</h4>
            <p class="text-sm text-blue-700">
              This file can be processed by the data pipeline. Files are automatically picked up by the pipeline when placed in a monitored directory.
            </p>
          </div>
        </div>
      </div>
    </div>
    """
  end

  # Helper functions to detect file types
  defp get_file_type(filename) when is_binary(filename) do
    cond do
      String.ends_with?(filename, ".csv") -> :csv
      String.ends_with?(filename, ".json") -> :json
      String.ends_with?(filename, ".xlsx") || String.ends_with?(filename, ".xls") -> :excel
      String.ends_with?(filename, ".xml") -> :xml
      String.ends_with?(filename, ".txt") -> :text
      true -> :unknown
    end
  end
  defp get_file_type(_), do: :unknown
  
  defp is_csv_file?(filename) when is_binary(filename) do
    String.ends_with?(filename, ".csv")
  end
  defp is_csv_file?(_), do: false

  defp is_json_file?(filename) when is_binary(filename) do
    String.ends_with?(filename, ".json")
  end
  defp is_json_file?(_), do: false

  defp is_valid_content?(content) do
    is_binary(content) && 
    !String.starts_with?(content, "File too large") && 
    !String.starts_with?(content, "Binary file") && 
    !String.starts_with?(content, "Error")
  end
end
