defmodule WebServiceWeb.Components.FileWatcher.BreadcrumbComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders a breadcrumb navigation component for the file watcher.
  
  ## Examples
  
      <.live_component
        module={WebServiceWeb.Components.FileWatcher.BreadcrumbComponent}
        id="breadcrumb"
        breadcrumbs={@breadcrumbs}
      />
  """
  def render(assigns) do
    ~H"""
    <nav class="flex" aria-label="Breadcrumb">
      <ol class="flex items-center space-x-1">
        <%= for {crumb, index} <- Enum.with_index(@breadcrumbs) do %>
          <li class="flex items-center">
            <%= if index > 0 do %>
              <svg class="mx-1 h-4 w-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
              </svg>
            <% end %>
            <button
              phx-click="navigate-path"
              phx-value-path={crumb.path}
              class="text-blue-600 hover:text-blue-800 font-medium truncate max-w-[150px] md:max-w-xs">
              <%= crumb.name %>
            </button>
          </li>
        <% end %>
      </ol>
    </nav>
    """
  end
end
