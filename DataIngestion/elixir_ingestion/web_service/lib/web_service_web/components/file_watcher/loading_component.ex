defmodule WebServiceWeb.Components.FileWatcher.LoadingComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders a loading spinner component for the file watcher.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.LoadingComponent}
        id="loading"
      />
  """
  def render(assigns) do
    ~H"""
    <div class="flex items-center justify-center py-20">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      <span class="ml-3 text-lg text-gray-600">Loading content...</span>
    </div>
    """
  end
end
