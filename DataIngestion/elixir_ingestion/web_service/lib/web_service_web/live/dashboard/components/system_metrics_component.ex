defmodule WebServiceWeb.DashboardComponents.SystemMetricsComponent do
  use Phoenix.Component

  attr :metrics, :map, required: true

  def system_metrics_component(assigns) do
    ~H"""
    <div class="bg-white rounded-lg shadow-md overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200">
        <h2 class="text-lg font-semibold text-gray-800">System Health</h2>
      </div>
      
      <div class="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <!-- Memory Usage -->
        <div class="bg-white rounded-lg border border-gray-200 p-4">
          <div class="flex justify-between items-start">
            <div>
              <h3 class="text-gray-500 text-sm font-medium">Memory Usage</h3>
              <div class="mt-1 flex items-baseline">
                <p class="text-2xl font-semibold text-gray-900">
                  <%= get_in(@metrics, [:memory, :percentage]) || "0" %>%
                </p>
                <p class="ml-2 text-sm text-gray-500">
                  <%= get_in(@metrics, [:memory, :used]) || "0" %> / <%= get_in(@metrics, [:memory, :total]) || "0" %> MB
                </p>
              </div>
            </div>
            <div class="bg-green-100 p-2 rounded-md">
              <svg class="h-5 w-5 text-green-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
          </div>
          <div class="mt-3">
            <div class="relative h-3 bg-gray-200 rounded-full overflow-hidden">
              <div class="absolute h-full bg-green-500 rounded-full" style={"width: #{get_in(@metrics, [:memory, :percentage]) || 0}%"}></div>
            </div>
          </div>
        </div>

        <!-- CPU Usage -->
        <div class="bg-white rounded-lg border border-gray-200 p-4">
          <div class="flex justify-between items-start">
            <div>
              <h3 class="text-gray-500 text-sm font-medium">CPU Usage</h3>
              <div class="mt-1 flex items-baseline">
                <p class="text-2xl font-semibold text-gray-900">
                  <%= get_in(@metrics, [:cpu, :percentage]) || "0" %>%
                </p>
              </div>
            </div>
            <div class="bg-blue-100 p-2 rounded-md">
              <svg class="h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
              </svg>
            </div>
          </div>
          <div class="mt-3">
            <div class="relative h-3 bg-gray-200 rounded-full overflow-hidden">
              <div class="absolute h-full bg-blue-500 rounded-full" style={"width: #{get_in(@metrics, [:cpu, :percentage]) || 0}%"}></div>
            </div>
          </div>
        </div>

        <!-- Disk Usage -->
        <div class="bg-white rounded-lg border border-gray-200 p-4">
          <div class="flex justify-between items-start">
            <div>
              <h3 class="text-gray-500 text-sm font-medium">Disk Usage</h3>
              <div class="mt-1 flex items-baseline">
                <p class="text-2xl font-semibold text-gray-900">
                  <%= get_in(@metrics, [:disk, :percentage]) || "0" %>%
                </p>
                <p class="ml-2 text-sm text-gray-500">
                  <%= get_in(@metrics, [:disk, :used]) || "0" %> / <%= get_in(@metrics, [:disk, :total]) || "0" %> GB
                </p>
              </div>
            </div>
            <div class="bg-purple-100 p-2 rounded-md">
              <svg class="h-5 w-5 text-purple-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
            </div>
          </div>
          <div class="mt-3">
            <div class="relative h-3 bg-gray-200 rounded-full overflow-hidden">
              <div class="absolute h-full bg-purple-500 rounded-full" style={"width: #{get_in(@metrics, [:disk, :percentage]) || 0}%"}></div>
            </div>
          </div>
        </div>

        <!-- Uptime -->
        <div class="bg-white rounded-lg border border-gray-200 p-4">
          <div class="flex justify-between items-start">
            <div>
              <h3 class="text-gray-500 text-sm font-medium">System Uptime</h3>
              <p class="text-2xl font-semibold text-gray-900">
                <%= format_uptime(get_in(@metrics, [:uptime, :seconds])) %>
              </p>
            </div>
            <div class="bg-yellow-100 p-2 rounded-md">
              <svg class="h-5 w-5 text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div class="mt-3 text-sm text-gray-500">
            Started: <%= get_in(@metrics, [:uptime, :started_at]) || "-" %>
          </div>
        </div>
      </div>
    </div>
    """
  end

  defp format_uptime(nil), do: "-"
  defp format_uptime(seconds) when is_integer(seconds) do
    days = div(seconds, 86400)
    hours = div(rem(seconds, 86400), 3600)
    minutes = div(rem(seconds, 3600), 60)
    
    cond do
      days > 0 -> "#{days}d #{hours}h #{minutes}m"
      hours > 0 -> "#{hours}h #{minutes}m"
      true -> "#{minutes}m"
    end
  end
  defp format_uptime(_), do: "-"
end
