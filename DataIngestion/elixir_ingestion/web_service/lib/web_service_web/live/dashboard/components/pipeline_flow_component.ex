defmodule WebServiceWeb.DashboardComponents.PipelineFlowComponent do
  use Phoenix.Component
  # Import only the raw function we need
  import Phoenix.HTML, only: [raw: 1]

  attr :pipeline_stages, :list, required: true

  def pipeline_flow_component(assigns) do
    ~H"""
    <div class="bg-white p-6 rounded-lg shadow-md">
      <h2 class="text-xl font-semibold mb-6">Pipeline Flow</h2>
      
      <div class="overflow-x-auto">
        <div class="flex items-center space-x-4 min-w-max pipeline-flow-container py-4">
          <%= for {stage, index} <- Enum.with_index(@pipeline_stages) do %>
            <!-- Stage node -->
            <div class="flex flex-col items-center">
              <div 
                class="relative w-28 h-20 rounded-lg flex flex-col items-center justify-center cursor-pointer shadow-sm transition-transform hover:shadow hover:-translate-y-1 duration-200"
                phx-click="show_stage_details"
                phx-value-stage={stage.name}
                style={get_stage_background_style(stage.status)}
              >
                <div class={get_stage_icon_class(stage.status)}>
                  <%= render_icon(stage.name) %>
                </div>
                <div class="text-xs font-medium mt-1"><%= stage.name %></div>
                
                <!-- Processing indicator dot -->
                <%= if stage.status == :active do %>
                  <div class="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-blue-500 animate-pulse"></div>
                <% end %>
                
                <!-- Status indicator at bottom -->
                <div class={"absolute bottom-0 left-0 right-0 h-1 rounded-b-lg #{get_stage_indicator_class(stage.status)}"}></div>
              </div>
              
              <!-- Stage name label -->
              <div class="text-xs text-gray-500 mt-2 text-center max-w-xs truncate">
                <%= stage.name %>
              </div>
            </div>
            
            <!-- Connector between stages -->
            <%= if index < length(@pipeline_stages) - 1 do %>
              <div class="connector-line flex items-center relative">
                <!-- Static connector line -->
                <div class="w-12 h-0.5 bg-gray-300"></div>
                
                <!-- Animated data flow indicator -->
                <%= if stage.status in [:active, :completed] do %>
                  <div class="data-flow absolute inset-0 flex items-center justify-center">
                    <i class="fas fa-chevron-right text-blue-500 data-flow-indicator"></i>
                  </div>
                <% end %>
              </div>
            <% end %>
          <% end %>
        </div>
      </div>
      
      <!-- Legend -->
      <div class="mt-4 flex flex-wrap gap-x-6 text-sm text-gray-600 justify-center">
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-gray-500 mr-2"></div>
          <span>Idle</span>
        </div>
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-blue-500 mr-2 animate-pulse"></div>
          <span>Active</span>
        </div>
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
          <span>Completed</span>
        </div>
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
          <span>Failed</span>
        </div>
      </div>
    </div>
    """
  end

  # Helper functions for stage styling
  defp get_stage_background_style(status) do
    case status do
      :idle -> "background-color: #f9fafb; border: 1px solid #e5e7eb;"
      :active -> "background-color: #eff6ff; border: 1px solid #93c5fd;"
      :completed -> "background-color: #f0fdf4; border: 1px solid #86efac;"
      :failed -> "background-color: #fef2f2; border: 1px solid #fca5a5;"
      _ -> "background-color: #f9fafb; border: 1px solid #e5e7eb;"
    end
  end
  
  defp get_stage_icon_class(status) do
    base_class = "text-xl"
    case status do
      :idle -> "#{base_class} text-gray-400"
      :active -> "#{base_class} text-blue-500"
      :completed -> "#{base_class} text-green-500"
      :failed -> "#{base_class} text-red-500"
      _ -> "#{base_class} text-gray-400"
    end
  end
  
  defp get_stage_indicator_class(status) do
    case status do
      :idle -> "bg-gray-300"
      :active -> "bg-blue-500"
      :completed -> "bg-green-500"
      :failed -> "bg-red-500"
      _ -> "bg-gray-300"
    end
  end
  
  defp render_icon(name) do
    icon_class = case name do
      "FileWatcher" -> "fas fa-folder-open"
      "Producer" -> "fas fa-stream"
      "Processor" -> "fas fa-cogs"
      "SchemaInference" -> "fas fa-table"
      "DataProfiler" -> "fas fa-chart-pie"
      "TimeSeriesProcessor" -> "fas fa-clock"
      "Validator" -> "fas fa-check-circle"
      "MetaDataEnricher" -> "fas fa-tags"
      "Transformer" -> "fas fa-exchange-alt"
      "Writer" -> "fas fa-pen-fancy"
      "TimescaleDB" -> "fas fa-database"
      _ -> "fas fa-cube"
    end
    
    raw("<i class=\"#{icon_class}\"></i>")
  end
end
