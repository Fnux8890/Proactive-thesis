defmodule WebServiceWeb.DashboardComponents.ActivityLogComponent do
  use Phoenix.Component

  attr :activity_logs, :list, required: true

  def activity_log_component(assigns) do
    ~H"""
    <div class="bg-white shadow rounded-lg p-6">
      <h2 class="text-xl font-semibold mb-6">Activity Log</h2>
      <div class="overflow-x-auto">
        <table class="min-w-full bg-white">
          <thead class="bg-gray-100">
            <tr>
              <th class="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
              <th class="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Component</th>
              <th class="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Level</th>
              <th class="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Message</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-200">
            <%= if Enum.empty?(@activity_logs) do %>
              <tr>
                <td colspan="4" class="py-8 px-4 text-center text-gray-500">
                  <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 class="mt-2 text-sm font-medium text-gray-900">No activity logs</h3>
                  <p class="mt-1 text-sm text-gray-500">Pipeline activity will appear here when processing begins.</p>
                </td>
              </tr>
            <% else %>
              <%= for log <- @activity_logs do %>
                <tr class="hover:bg-gray-50">
                  <td class="py-4 px-4 text-sm text-gray-900 font-medium"><%= format_datetime(log.timestamp) %></td>
                  <td class="py-4 px-4 text-sm text-gray-500"><%= log.component %></td>
                  <td class="py-4 px-4 text-sm">
                    <span class={get_log_level_class(log.level)}>
                      <%= String.upcase(to_string(log.level)) %>
                    </span>
                  </td>
                  <td class="py-4 px-4 text-sm text-gray-500"><%= log.message %></td>
                </tr>
              <% end %>
            <% end %>
          </tbody>
        </table>
      </div>
    </div>
    """
  end

  # Helper functions
  defp get_log_level_class(level) do
    case level do
      :info -> "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
      :warning -> "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800"
      :error -> "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800"
      _ -> "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
    end
  end

  # Time formatting helper
  defp format_datetime(nil), do: "-"
  defp format_datetime(datetime) when is_binary(datetime) do
    case DateTime.from_iso8601(datetime) do
      {:ok, dt, _} -> format_datetime(dt)
      _ -> datetime
    end
  end
  defp format_datetime(datetime) do
    datetime
    |> Calendar.strftime("%Y-%m-%d %H:%M:%S UTC")
  end

  # Format relative time (e.g., "2 minutes ago")
  # Commented out as currently unused - will be used in future UI enhancements
  # defp format_relative_time(datetime) do
  #   diff_seconds = DateTime.diff(DateTime.utc_now(), datetime, :second)
  #   
  #   cond do
  #     diff_seconds < 60 -> "just now"
  #     diff_seconds < 3600 -> "#{div(diff_seconds, 60)} min ago"
  #     diff_seconds < 86400 -> "#{div(diff_seconds, 3600)} hrs ago"
  #     true -> "#{div(diff_seconds, 86400)} days ago"
  #   end
  # end
end
