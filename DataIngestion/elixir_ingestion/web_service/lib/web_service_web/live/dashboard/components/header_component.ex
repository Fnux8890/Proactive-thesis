defmodule WebServiceWeb.DashboardComponents.HeaderComponent do
  use Phoenix.Component

  attr :last_refresh, :any, required: true
  attr :current_time, :any, required: true

  def header_component(assigns) do
    ~H"""
    <div class="p-6 bg-white border-b border-gray-200 shadow-sm mb-6 rounded-lg">
      <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-800">Data Pipeline Dashboard</h1>
          <p class="mt-2 text-gray-600">
            Real-time monitoring for the Elixir data ingestion pipeline
          </p>
        </div>
        <div class="mt-4 sm:mt-0 flex flex-col sm:flex-row items-end space-y-2 sm:space-y-0 sm:space-x-4">
          <div class="text-sm text-gray-500 whitespace-nowrap">
            Last refreshed: <%= format_datetime(@last_refresh) %>
          </div>
          <button phx-click="manual_refresh" class="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>
      </div>
    </div>
    """
  end

  # Helper function for formatting dates
  def format_datetime(nil), do: "-"
  def format_datetime(datetime) when is_binary(datetime) do
    case DateTime.from_iso8601(datetime) do
      {:ok, dt, _} -> format_datetime(dt)
      _ -> datetime
    end
  end
  def format_datetime(datetime) do
    datetime
    |> Calendar.strftime("%Y-%m-%d %H:%M:%S UTC")
  end
end
