defmodule WebServiceWeb.Components.FileWatcher.ViewToggleComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders a view mode toggle (grid/list) component for the file watcher UI.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.ViewToggleComponent}
        id="view_toggle"
        view_mode={@view_mode}
      />
  """
  def render(assigns) do
    ~H"""
    <div class="hidden md:flex items-center bg-gray-100 rounded-lg p-1">
      <button phx-click="set-view-mode" phx-value-mode="grid" class={"px-3 py-1 rounded-md transition-all #{if @view_mode == "grid", do: "bg-white shadow-sm", else: "hover:bg-gray-200"}"}>
        <i class="fas fa-th text-gray-600"></i>
      </button>
      <button phx-click="set-view-mode" phx-value-mode="list" class={"px-3 py-1 rounded-md transition-all #{if @view_mode != "grid", do: "bg-white shadow-sm", else: "hover:bg-gray-200"}"}>
        <i class="fas fa-list text-gray-600"></i>
      </button>
    </div>
    """
  end
end
