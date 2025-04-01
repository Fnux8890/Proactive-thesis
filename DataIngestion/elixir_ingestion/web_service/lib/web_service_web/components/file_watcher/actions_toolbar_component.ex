defmodule WebServiceWeb.Components.FileWatcher.ActionsToolbarComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders an actions toolbar component for file operations in the file watcher UI.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.ActionsToolbarComponent}
        id="actions_toolbar"
        current_path={@current_path}
        current_file={@current_file}
      />
  """
  def render(assigns) do
    ~H"""
    <div class="bg-white border-b border-gray-200 p-2 flex items-center justify-between">
      <div class="flex space-x-2">
        <button
          title="Upload File"
          class="p-2 rounded-lg hover:bg-gray-100 text-gray-700 transition-colors"
          phx-click="show-upload-modal"
        >
          <i class="fas fa-upload mr-1"></i>
          <span class="hidden sm:inline">Upload</span>
        </button>
        
        <button
          title="New Folder"
          class="p-2 rounded-lg hover:bg-gray-100 text-gray-700 transition-colors"
          phx-click="show-new-folder-modal"
        >
          <i class="fas fa-folder-plus mr-1"></i>
          <span class="hidden sm:inline">New Folder</span>
        </button>
      </div>
      
      <div class="flex space-x-2">
        <%= if @current_file do %>
          <button
            title="Download File"
            class="p-2 rounded-lg hover:bg-gray-100 text-gray-700 transition-colors"
            phx-click="download-file"
            phx-value-file={@current_file}
          >
            <i class="fas fa-download mr-1"></i>
            <span class="hidden sm:inline">Download</span>
          </button>
          
          <button
            title="Delete File"
            class="p-2 rounded-lg hover:bg-red-50 text-red-600 transition-colors"
            phx-click="show-delete-modal"
            phx-value-file={@current_file}
          >
            <i class="fas fa-trash-alt mr-1"></i>
            <span class="hidden sm:inline">Delete</span>
          </button>
        <% end %>
        
        <div class="border-l border-gray-200 h-6 mx-1"></div>
        
        <button
          title="Refresh Directory"
          class="p-2 rounded-lg hover:bg-gray-100 text-gray-700 transition-colors"
          phx-click="refresh-files"
        >
          <i class="fas fa-sync-alt mr-1"></i>
          <span class="hidden sm:inline">Refresh</span>
        </button>
      </div>
    </div>
    """
  end
end
