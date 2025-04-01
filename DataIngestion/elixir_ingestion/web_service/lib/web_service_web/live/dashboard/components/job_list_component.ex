defmodule WebServiceWeb.DashboardComponents.JobListComponent do
  use Phoenix.Component

  attr :jobs, :list, required: true
  attr :filtered_status, :atom, required: true

  def job_list_component(assigns) do
    ~H"""
    <div class="bg-white rounded-lg shadow-md overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
        <h2 class="text-lg font-semibold text-gray-800">Job History</h2>
        
        <!-- Filter buttons -->
        <div class="flex space-x-2">
          <button phx-click="filter_jobs" phx-value-status="all" class={"px-3 py-1 rounded-full text-sm transition #{if @filtered_status == :all, do: "bg-blue-100 text-blue-700", else: "bg-gray-200 text-gray-700 hover:bg-gray-300"}"}>
            All
          </button>
          <button phx-click="filter_jobs" phx-value-status="completed" class={"px-3 py-1 rounded-full text-sm transition #{if @filtered_status == :completed, do: "bg-green-100 text-green-700", else: "bg-gray-200 text-gray-700 hover:bg-gray-300"}"}>
            Completed
          </button>
          <button phx-click="filter_jobs" phx-value-status="processing" class={"px-3 py-1 rounded-full text-sm transition #{if @filtered_status == :processing, do: "bg-yellow-100 text-yellow-700", else: "bg-gray-200 text-gray-700 hover:bg-gray-300"}"}>
            Processing
          </button>
          <button phx-click="filter_jobs" phx-value-status="failed" class={"px-3 py-1 rounded-full text-sm transition #{if @filtered_status == :failed, do: "bg-red-100 text-red-700", else: "bg-gray-200 text-gray-700 hover:bg-gray-300"}"}>
            Failed
          </button>
        </div>
      </div>
      
      <%= if Enum.empty?(@jobs) do %>
        <div class="p-6 text-center text-gray-500">
          <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
          </svg>
          <h3 class="mt-2 text-sm font-medium text-gray-900">No jobs found</h3>
          <p class="mt-1 text-sm text-gray-500">
            <%= case @filtered_status do %>
              <% :all -> %>
                No jobs have been processed yet.
              <% :completed -> %>
                No completed jobs found.
              <% :processing -> %>
                No jobs are currently being processed.
              <% :failed -> %>
                No failed jobs found.
              <% _ -> %>
                No jobs match the current filter.
            <% end %>
          </p>
        </div>
      <% else %>
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  ID
                </th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Source
                </th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Start Time
                </th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Duration
                </th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th scope="col" class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <%= for job <- @jobs do %>
                <tr class="hover:bg-gray-50">
                  <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    <%= job.id %>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <%= job.source %>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <%= format_datetime(job.start_time) %>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <%= format_duration(job.duration) %>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <span class={status_badge_class(job.status)}>
                      <%= String.capitalize(to_string(job.status)) %>
                    </span>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button phx-click="view_job_details" phx-value-job-id={job.id} class="text-blue-600 hover:text-blue-900">
                      Details
                    </button>
                  </td>
                </tr>
              <% end %>
            </tbody>
          </table>
        </div>
      <% end %>
    </div>
    """
  end

  # Helper functions
  defp status_badge_class(status) do
    base_classes = "px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
    
    status_specific = case status do
      :completed -> "bg-green-100 text-green-800"
      :processing -> "bg-yellow-100 text-yellow-800"
      :failed -> "bg-red-100 text-red-800"
      _ -> "bg-gray-100 text-gray-800"
    end
    
    "#{base_classes} #{status_specific}"
  end

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

  defp format_duration(nil), do: "-"
  defp format_duration(duration) when is_integer(duration) do
    cond do
      duration < 1000 -> "#{duration}ms"
      duration < 60_000 -> "#{Float.round(duration / 1000, 1)}s"
      duration < 3_600_000 -> "#{div(duration, 60_000)}m #{div(rem(duration, 60_000), 1000)}s"
      true -> "#{div(duration, 3_600_000)}h #{div(rem(duration, 3_600_000), 60_000)}m"
    end
  end
  defp format_duration(duration), do: "#{duration}"
end
