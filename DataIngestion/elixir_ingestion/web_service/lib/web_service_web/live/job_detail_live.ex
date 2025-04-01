defmodule WebServiceWeb.JobDetailLive do
  use Phoenix.LiveView
  alias WebService.ResultsMonitor
  import Phoenix.Component
  # Path helpers
  use WebServiceWeb, :verified_routes

  def mount(%{"id" => id}, _session, socket) do
    # Parse job ID
    job_id = case Integer.parse(id) do
      {num, ""} -> num
      _ -> nil
    end
    
    # Get job details
    job = if job_id, do: ResultsMonitor.get_job(job_id)
    
    if connected?(socket) do
      Phoenix.PubSub.subscribe(WebService.PubSub, "jobs")
    end
    
    socket = assign(socket, 
      job: job,
      job_id: job_id,
      page_title: job && "Job ##{job_id}" || "Job Not Found"
    )
    
    {:ok, socket}
  end
  
  def handle_info({:job_updated, updated_job}, socket) do
    # Only update if this is the job we're viewing
    if updated_job.id == socket.assigns.job_id do
      {:noreply, assign(socket, job: updated_job)}
    else
      {:noreply, socket}
    end
  end
  
  def render(assigns) do
    ~H"""
    <div class="container mx-auto px-4 py-8">
      <div class="mb-6">
        <.link navigate={~p"/"} class="text-blue-500 hover:underline">
          &larr; Back to Dashboard
        </.link>
      </div>
      
      <%= if @job do %>
        <h1 class="text-3xl font-bold mb-6">Job #<%= @job.id %>: <%= Path.basename(@job.file_path) %></h1>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div class="bg-white shadow rounded-lg p-6">
            <h2 class="text-xl font-semibold mb-4">Job Information</h2>
            
            <div class="space-y-4">
              <div>
                <div class="text-sm font-medium text-gray-500">Status</div>
                <div class="mt-1">
                  <span class={status_badge_class(@job.status)}>
                    <%= String.upcase(to_string(@job.status)) %>
                  </span>
                </div>
              </div>
              
              <div>
                <div class="text-sm font-medium text-gray-500">File</div>
                <div class="mt-1 text-sm text-gray-900 break-all">
                  <%= @job.file_path %>
                </div>
              </div>
              
              <div>
                <div class="text-sm font-medium text-gray-500">Started At</div>
                <div class="mt-1 text-sm text-gray-900">
                  <%= format_datetime(@job.started_at) %>
                </div>
              </div>
              
              <div>
                <div class="text-sm font-medium text-gray-500">Completed At</div>
                <div class="mt-1 text-sm text-gray-900">
                  <%= format_datetime(@job.completed_at) %>
                </div>
              </div>
            </div>
          </div>
          
          <div class="bg-white shadow rounded-lg p-6">
            <h2 class="text-xl font-semibold mb-4">Data File Analysis</h2>
            
            <%= if @job.details do %>
              <%= if get_in(@job.details, ["data", "file_size_bytes"]) do %>
                <div class="space-y-4">
                  <div>
                    <div class="text-sm font-medium text-gray-500">File Size</div>
                    <div class="mt-1 text-sm text-gray-900">
                      <%= format_bytes(get_in(@job.details, ["data", "file_size_bytes"])) %>
                    </div>
                  </div>
                  
                  <%= if get_in(@job.details, ["data", "line_count"]) do %>
                    <div>
                      <div class="text-sm font-medium text-gray-500">Line Count</div>
                      <div class="mt-1 text-sm text-gray-900">
                        <%= get_in(@job.details, ["data", "line_count"]) %>
                      </div>
                    </div>
                  <% end %>
                  
                  <%= if get_in(@job.details, ["data", "headers"]) do %>
                    <div>
                      <div class="text-sm font-medium text-gray-500">Headers</div>
                      <div class="mt-1 text-sm text-gray-900">
                        <%= inspect(get_in(@job.details, ["data", "headers"])) %>
                      </div>
                    </div>
                  <% end %>
                  
                  <%= if get_in(@job.details, ["data", "delimiter"]) do %>
                    <div>
                      <div class="text-sm font-medium text-gray-500">Delimiter</div>
                      <div class="mt-1 text-sm text-gray-900">
                        <%= if get_in(@job.details, ["data", "delimiter"]) == ";" do %>
                          Semicolon (;)
                        <% else %>
                          Comma (,)
                        <% end %>
                      </div>
                    </div>
                  <% end %>
                </div>
              <% else %>
                <div class="p-4 text-sm text-gray-500">
                  No detailed analysis available.
                </div>
              <% end %>
            <% else %>
              <div class="p-4 text-sm text-gray-500">
                No details available for this job.
              </div>
            <% end %>
          </div>
        </div>
        
        <%= if get_in(@job.details, ["data", "sample"]) do %>
          <div class="bg-white shadow rounded-lg p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Data Sample</h2>
            <div class="overflow-x-auto">
              <pre class="bg-gray-800 text-gray-100 p-4 rounded text-sm"><%= Jason.encode!(get_in(@job.details, ["data", "sample"]), pretty: true) %></pre>
            </div>
          </div>
        <% end %>
        
        <%= if @job.details do %>
          <div class="bg-white shadow rounded-lg p-6">
            <h2 class="text-xl font-semibold mb-4">Raw Job Result</h2>
            <div class="overflow-x-auto">
              <pre class="bg-gray-800 text-gray-100 p-4 rounded text-sm"><%= Jason.encode!(@job.details, pretty: true) %></pre>
            </div>
          </div>
        <% end %>
        
      <% else %>
        <div class="bg-red-50 p-6 rounded-lg border border-red-200">
          <h1 class="text-2xl font-bold text-red-800 mb-2">Job Not Found</h1>
          <p class="text-red-600">No job found with ID: <%= @job_id %></p>
        </div>
      <% end %>
    </div>
    """
  end
  
  # Helper functions
  
  defp status_badge_class(status) do
    base_classes = "px-2 py-1 rounded text-xs font-medium"
    
    status_specific = case status do
      :pending -> "bg-yellow-100 text-yellow-800"
      :processing -> "bg-blue-100 text-blue-800"
      :completed -> "bg-green-100 text-green-800"
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
    Calendar.strftime(datetime, "%Y-%m-%d %H:%M:%S")
  end
  
  defp format_bytes(bytes) when is_binary(bytes) do
    case Integer.parse(bytes) do
      {num, _} -> format_bytes(num)
      _ -> "#{bytes} bytes"
    end
  end
  defp format_bytes(bytes) when is_number(bytes) do
    cond do
      bytes < 1024 -> "#{bytes} bytes"
      bytes < 1024 * 1024 -> "#{Float.round(bytes / 1024, 2)} KB"
      bytes < 1024 * 1024 * 1024 -> "#{Float.round(bytes / 1024 / 1024, 2)} MB"
      true -> "#{Float.round(bytes / 1024 / 1024 / 1024, 2)} GB"
    end
  end
  defp format_bytes(_), do: "Unknown size"
end
