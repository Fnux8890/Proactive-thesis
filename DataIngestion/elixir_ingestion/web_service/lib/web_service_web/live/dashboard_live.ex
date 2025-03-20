defmodule WebServiceWeb.DashboardLive do
  use WebServiceWeb, :live_view
  
  alias WebService.ResultsMonitor
  alias WebServiceWeb.DashboardComponents.{
    HeaderComponent,
    StatusCardsComponent,
    PipelineFlowComponent,
    ActivityLogComponent,
    JobListComponent,
    SystemMetricsComponent
  }

  # Import the function components
  import WebServiceWeb.DashboardComponents.HeaderComponent
  import WebServiceWeb.DashboardComponents.StatusCardsComponent
  import WebServiceWeb.DashboardComponents.PipelineFlowComponent
  import WebServiceWeb.DashboardComponents.ActivityLogComponent
  import WebServiceWeb.DashboardComponents.JobListComponent
  import WebServiceWeb.DashboardComponents.SystemMetricsComponent

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Set up timer to refresh data every 10 seconds
      :timer.send_interval(10_000, self(), :refresh_data)
    end

    jobs = ResultsMonitor.get_all_jobs()
    status_counts = get_status_counts(jobs)
    pipeline_stages = get_pipeline_stages()
    
    # Generate sample activity logs - in a real application, these would come from your logging system
    activity_logs = get_sample_activity_logs()
    
    # Generate system metrics - in a real application, these would come from system monitoring
    system_metrics = get_system_metrics()

    {:ok, assign(socket,
      jobs: jobs,
      status_counts: status_counts,
      pipeline_stages: pipeline_stages,
      current_time: DateTime.utc_now(),
      page_title: "Data Pipeline Dashboard",
      show_stage_modal: false,
      selected_stage: nil,
      stage_description: nil,
      stage_metrics: nil,
      filtered_status: :all,
      last_refresh: DateTime.utc_now(),
      activity_logs: activity_logs,
      system_metrics: system_metrics
    )}
  end

  @impl true
  def handle_event("filter_jobs", %{"status" => status_string}, socket) do
    status = String.to_existing_atom(status_string)
    filtered_jobs = filter_jobs(socket.assigns.jobs, status)
    
    {:noreply, 
     socket
     |> assign(filtered_status: status)
     |> assign(jobs: filtered_jobs)}
  end

  @impl true
  def handle_event("show_stage_details", %{"stage" => stage_name}, socket) do
    # Here we would fetch real details about the pipeline stage
    # For demo, we'll use static descriptions
    {description, metrics} = get_stage_details(stage_name)
    
    {:noreply, 
     socket
     |> assign(show_stage_modal: true)
     |> assign(selected_stage: stage_name)
     |> assign(stage_description: description)
     |> assign(stage_metrics: metrics)}
  end

  @impl true
  def handle_event("close_modal", _, socket) do
    {:noreply, assign(socket, show_stage_modal: false)}
  end

  @impl true
  def handle_event("manual_refresh", _, socket) do
    # Here we would fetch fresh data
    send(self(), :refresh_data)
    {:noreply, socket}
  end

  @impl true
  def handle_info(:refresh_data, socket) do
    # In a real app, this would fetch fresh data from your monitoring services
    jobs = ResultsMonitor.get_all_jobs()
    status_counts = get_status_counts(jobs)
    
    # Update pipeline stages with new status
    pipeline_stages = get_pipeline_stages() 
    |> Enum.map(fn stage -> 
      # Update stage status based on job processing
      # This is a simple simulation - in a real app you'd get actual statuses
      %{stage | status: get_random_stage_status()}
    end)
    
    # Update system metrics
    system_metrics = get_system_metrics()
    
    {:noreply, 
     socket
     |> assign(jobs: filter_jobs(jobs, socket.assigns.filtered_status))
     |> assign(status_counts: status_counts)
     |> assign(pipeline_stages: pipeline_stages)
     |> assign(system_metrics: system_metrics)
     |> assign(last_refresh: DateTime.utc_now())}
  end

  # Helper functions for data management
  
  defp get_status_counts(jobs) do
    completed = Enum.count(jobs, &(&1.status == :completed))
    processing = Enum.count(jobs, &(&1.status == :processing))
    failed = Enum.count(jobs, &(&1.status == :failed))
    
    %{
      total: length(jobs),
      completed: completed,
      processing: processing,
      failed: failed
    }
  end
  
  defp get_pipeline_stages do
    [
      %{name: "FileWatcher", count: 0, status: :idle, icon: "heroicons-outline:folder-open"},
      %{name: "Producer", count: 0, status: :idle, icon: "heroicons-outline:collection"},
      %{name: "Processor", count: 0, status: :idle, icon: "heroicons-outline:cog"},
      %{name: "SchemaInference", count: 0, status: :idle, icon: "heroicons-outline:table"},
      %{name: "DataProfiler", count: 0, status: :idle, icon: "heroicons-outline:chart-bar"},
      %{name: "TimeSeriesProcessor", count: 0, status: :idle, icon: "heroicons-outline:clock"},
      %{name: "Validator", count: 0, status: :idle, icon: "heroicons-outline:check-circle"},
      %{name: "MetaDataEnricher", count: 0, status: :idle, icon: "heroicons-outline:tag"},
      %{name: "Transformer", count: 0, status: :idle, icon: "heroicons-outline:refresh"},
      %{name: "Writer", count: 0, status: :idle, icon: "heroicons-outline:pencil-alt"},
      %{name: "TimescaleDB", count: 0, status: :idle, icon: "heroicons-outline:database"}
    ]
  end
  
  defp get_sample_activity_logs do
    [
      %{
        id: "log-1",
        timestamp: DateTime.add(DateTime.utc_now(), -2, :minute),
        level: :info, 
        component: "FileWatcher",
        message: "Detected new CSV file 'sales_data_2023.csv'"
      },
      %{
        id: "log-2",
        timestamp: DateTime.add(DateTime.utc_now(), -5, :minute),
        level: :info,
        component: "Producer",
        message: "Started processing job #1234 for file 'sales_data_2023.csv'"
      },
      %{
        id: "log-3",
        timestamp: DateTime.add(DateTime.utc_now(), -15, :minute),
        level: :warning,
        component: "SchemaInference",
        message: "Ambiguous column type detected for 'customer_id'"
      },
      %{
        id: "log-4",
        timestamp: DateTime.add(DateTime.utc_now(), -30, :minute),
        level: :error,
        component: "Validator", 
        message: "Failed to validate row 123: missing required field 'timestamp'"
      }
    ]
  end
  
  defp get_system_metrics do
    # In a real app, these would come from actual system monitoring
    # Using placeholder data for demonstration
    %{
      memory: %{
        used: 1024,
        total: 4096,
        percentage: 25
      },
      cpu: %{
        percentage: 30
      },
      disk: %{
        used: 42,
        total: 100,
        percentage: 42
      },
      uptime: %{
        seconds: 86400 * 3 + 3600 * 5 + 60 * 15, # 3 days, 5 hours, 15 minutes
        started_at: "2025-03-08 10:00:00 UTC"
      }
    }
  end
  
  defp get_random_stage_status do
    # This is just for demo purposes - in a real app you'd get actual status
    Enum.random([:idle, :active, :completed, :failed])
  end
  
  defp get_stage_details(stage_name) do
    # This would typically fetch real data about the stage from your system
    # For demo purposes, we return static descriptions
    description = case stage_name do
      "FileWatcher" -> "Monitors file systems for new data files. Supports CSV, JSON, and XML formats."
      "Producer" -> "Creates data processing jobs and distributes work to appropriate processors."
      "Processor" -> "Type-specific processor that handles different file formats and data structures."
      "SchemaInference" -> "Analyzes data structure to infer schema and data types."
      "DataProfiler" -> "Analyzes data to detect patterns, outliers, and statistical properties."
      "TimeSeriesProcessor" -> "Specialized processor for time-series data, handling temporal aspects."
      "Validator" -> "Validates data against schema and business rules."
      "MetaDataEnricher" -> "Adds metadata and context to the processed data."
      "Transformer" -> "Transforms data into the format required by the genetic algorithm."
      "Writer" -> "Writes processed data to the persistent storage."
      "TimescaleDB" -> "Time-series optimized database for storing processed data."
      _ -> "Unknown pipeline stage"
    end
    
    # Sample metrics for the selected stage
    metrics = %{
      processed_items: Enum.random(1000..5000),
      average_processing_time: Enum.random(50..200),
      error_rate: Float.round(Enum.random(0..50) / 1000, 4),
      throughput: Float.round(Enum.random(50..200) / 10, 1)
    }
    
    {description, metrics}
  end
  
  defp filter_jobs(jobs, :all), do: jobs
  defp filter_jobs(jobs, status), do: Enum.filter(jobs, fn job -> job.status == status end)

  # Template rendering

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-gray-50">
      <div class="container mx-auto px-4 py-8">
        <!-- Header Component -->
        <.header_component 
          last_refresh={@last_refresh}
          current_time={@current_time}
        />
        
        <!-- Status Overview Cards Component -->
        <.status_cards_component status_counts={@status_counts} />
        
        <!-- System Metrics Component -->
        <div class="mb-8">
          <.system_metrics_component metrics={@system_metrics} />
        </div>
        
        <!-- Pipeline Flow Component -->
        <div class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
          <div class="col-span-1 md:col-span-5">
            <.pipeline_flow_component pipeline_stages={@pipeline_stages} />
          </div>
        </div>
        
        <!-- Main content area with 2 column layout for wider screens -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Jobs table - takes 2/3 of width on large screens -->
          <div class="lg:col-span-2">
            <.job_list_component jobs={@jobs} filtered_status={@filtered_status} />
          </div>
          
          <!-- Activity log - takes 1/3 of width on large screens -->
          <div>
            <.activity_log_component activity_logs={@activity_logs} />
          </div>
        </div>
      </div>
      
      <!-- Stage Details Modal -->
      <%= if @show_stage_modal do %>
        <div class="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
          <div class="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 class="text-lg font-semibold text-gray-900"><%= @selected_stage %> Details</h3>
              <button class="text-gray-400 hover:text-gray-500" phx-click="close_modal">
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div class="px-6 py-4">
              <h4 class="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Description</h4>
              <p class="text-gray-700 mb-4"><%= @stage_description %></p>
              
              <h4 class="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2 mt-6">Performance Metrics</h4>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div class="bg-gray-50 p-4 rounded-lg">
                  <p class="text-sm text-gray-500">Items Processed</p>
                  <p class="text-2xl font-semibold text-gray-800"><%= @stage_metrics.processed_items %></p>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                  <p class="text-sm text-gray-500">Avg. Processing Time</p>
                  <p class="text-2xl font-semibold text-gray-800"><%= @stage_metrics.average_processing_time %> ms</p>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                  <p class="text-sm text-gray-500">Error Rate</p>
                  <p class="text-2xl font-semibold text-gray-800"><%= @stage_metrics.error_rate * 100 %>%</p>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                  <p class="text-sm text-gray-500">Throughput</p>
                  <p class="text-2xl font-semibold text-gray-800"><%= @stage_metrics.throughput %> items/sec</p>
                </div>
              </div>
            </div>
            <div class="px-6 py-4 border-t border-gray-200 flex justify-end">
              <button class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700" phx-click="close_modal">Close</button>
            </div>
          </div>
        </div>
      <% end %>
    </div>
    
    <style>
      /* Pipeline flow animation */
      @keyframes flow {
        0% { opacity: 0; transform: translateX(-4px); }
        50% { opacity: 1; }
        100% { opacity: 0; transform: translateX(4px); }
      }
      
      .data-flow-indicator {
        animation: flow 1.2s infinite;
      }
      
      .pipeline-flow-container {
        scrollbar-width: thin;
        scrollbar-color: #cbd5e1 #f1f5f9;
      }
      
      .pipeline-flow-container::-webkit-scrollbar {
        height: 6px;
      }
      
      .pipeline-flow-container::-webkit-scrollbar-track {
        background: #f1f5f9;
      }
      
      .pipeline-flow-container::-webkit-scrollbar-thumb {
        background-color: #cbd5e1;
        border-radius: 3px;
      }
    </style>
    """
  end
end
