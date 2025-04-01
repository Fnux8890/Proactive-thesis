defmodule WebServiceWeb.Components.FileWatcher.EmptyStateComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders an empty state component for the file watcher when a directory has no contents.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.EmptyStateComponent}
        id="empty_state"
      />
  """
  def render(assigns) do
    ~H"""
    <div class="py-16 flex flex-col items-center justify-center text-center bg-white rounded-lg border border-dashed border-gray-300">
      <div class="bg-gray-100 p-4 rounded-full mb-4">
        <i class="fas fa-folder-open text-gray-400 text-3xl"></i>
      </div>
      <h3 class="text-lg font-medium text-gray-900 mb-1">This directory is empty</h3>
      <p class="text-gray-500 max-w-md mb-6">
        Upload files here or create a new subdirectory to organize your data for the pipeline processing.
      </p>
    </div>
    """
  end
end
