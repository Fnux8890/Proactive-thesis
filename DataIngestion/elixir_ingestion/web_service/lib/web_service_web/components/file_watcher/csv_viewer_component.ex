defmodule WebServiceWeb.Components.FileWatcher.CsvViewerComponent do
  use WebServiceWeb, :live_component

  @doc """
  Renders a CSV file as an interactive data table.

  ## Examples

      <.live_component
        module={WebServiceWeb.Components.FileWatcher.CsvViewerComponent}
        id="csv_viewer"
        csv_data={@csv_data}
        file_name={@current_file}
      />
  """
  def mount(socket) do
    {:ok,
     socket
     |> assign(:page, 1)
     |> assign(:rows_per_page, 25)
     |> assign(:current_sort_column, nil)
     |> assign(:sort_direction, :asc)
     |> assign(:search_term, "")
     |> assign(:column_search, %{})}
  end

  def update(assigns, socket) do
    socket = assign(socket, assigns)

    # Parse CSV data if available
    socket =
      if Map.has_key?(socket.assigns, :csv_data) && is_binary(socket.assigns.csv_data) do
        {headers, rows} = parse_csv(socket.assigns.csv_data)

        socket
        |> assign(:headers, headers)
        |> assign(:rows, rows)
        |> assign(:filtered_rows, rows)
        |> assign(:total_pages, max(ceil(length(rows) / socket.assigns.rows_per_page), 1))
      else
        socket
        |> assign(:headers, [])
        |> assign(:rows, [])
        |> assign(:filtered_rows, [])
        |> assign(:total_pages, 1)
      end

    {:ok, socket}
  end

  def handle_event("change-page", %{"page" => page}, socket) do
    page = String.to_integer(page)

    {:noreply, assign(socket, :page, page)}
  end

  def handle_event("sort", %{"column" => column}, socket) do
    column_index = String.to_integer(column)

    {new_direction, new_column} =
      if socket.assigns.current_sort_column == column_index do
        if socket.assigns.sort_direction == :asc, do: {:desc, column_index}, else: {:asc, column_index}
      else
        {:asc, column_index}
      end

    sorted_rows =
      socket.assigns.filtered_rows
      |> Enum.sort_by(fn row ->
        if length(row) > column_index, do: Enum.at(row, column_index), else: ""
      end, (if new_direction == :asc, do: :asc, else: :desc))

    {:noreply,
     socket
     |> assign(:filtered_rows, sorted_rows)
     |> assign(:current_sort_column, new_column)
     |> assign(:sort_direction, new_direction)
     |> assign(:page, 1)}
  end

  def handle_event("search", %{"search" => %{"term" => term}}, socket) do
    filtered_rows =
      if term && term != "" do
        term = String.downcase(term)

        socket.assigns.rows
        |> Enum.filter(fn row ->
          Enum.any?(row, fn cell ->
            cell && String.downcase(cell) =~ term
          end)
        end)
      else
        socket.assigns.rows
      end

    {:noreply,
     socket
     |> assign(:search_term, term)
     |> assign(:filtered_rows, filtered_rows)
     |> assign(:total_pages, max(ceil(length(filtered_rows) / socket.assigns.rows_per_page), 1))
     |> assign(:page, 1)}
  end

  def handle_event("column-search", %{"column" => column, "value" => value}, socket) do
    column_index = String.to_integer(column)

    column_search =
      if value && value != "" do
        Map.put(socket.assigns.column_search, column_index, value)
      else
        Map.delete(socket.assigns.column_search, column_index)
      end

    filtered_rows = apply_column_filters(socket.assigns.rows, column_search)

    {:noreply,
     socket
     |> assign(:column_search, column_search)
     |> assign(:filtered_rows, filtered_rows)
     |> assign(:total_pages, max(ceil(length(filtered_rows) / socket.assigns.rows_per_page), 1))
     |> assign(:page, 1)}
  end

  def handle_event("change-rows-per-page", %{"rows" => rows}, socket) do
    rows_per_page = String.to_integer(rows)

    {:noreply,
     socket
     |> assign(:rows_per_page, rows_per_page)
     |> assign(:total_pages, max(ceil(length(socket.assigns.filtered_rows) / rows_per_page), 1))
     |> assign(:page, 1)}
  end

  def render(assigns) do
    ~H"""
    <div class="bg-white rounded-lg border border-gray-200 overflow-hidden mt-4">
      <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center px-4 py-3 bg-gray-50 border-b border-gray-200">
        <h3 class="text-lg font-semibold text-gray-800 flex items-center mb-2 sm:mb-0">
          <i class="fas fa-table mr-2 text-green-600"></i>
          <%= @file_name %>
        </h3>

        <div class="flex flex-col sm:flex-row items-center gap-2">
          <form phx-submit="search" phx-target={@myself} class="relative">
            <input
              type="text"
              name="search[term]"
              value={@search_term}
              placeholder="Search..."
              class="py-1 pl-8 pr-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <i class="fas fa-search absolute left-2.5 top-2 text-gray-400 text-sm"></i>
          </form>

          <select
            phx-change="change-rows-per-page"
            phx-target={@myself}
            name="rows"
            class="text-sm border border-gray-300 rounded-md p-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="10" selected={@rows_per_page == 10}>10 rows</option>
            <option value="25" selected={@rows_per_page == 25}>25 rows</option>
            <option value="50" selected={@rows_per_page == 50}>50 rows</option>
            <option value="100" selected={@rows_per_page == 100}>100 rows</option>
          </select>

          <div class="flex items-center space-x-1">
            <button
              phx-click="export-csv"
              phx-target={@myself}
              class="bg-blue-50 hover:bg-blue-100 text-blue-700 text-sm py-1 px-2 rounded-md flex items-center"
            >
              <i class="fas fa-download mr-1"></i>
              Export
            </button>
          </div>
        </div>
      </div>

      <!-- CSV Table View -->
      <div class="overflow-x-auto">
        <%= if @headers && length(@headers) > 0 do %>
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <%= for {header, index} <- Enum.with_index(@headers) do %>
                  <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    <div class="flex flex-col">
                      <div class="flex items-center cursor-pointer group" phx-click="sort" phx-value-column={index} phx-target={@myself}>
                        <%= header %>
                        <div class="ml-1">
                          <%= if @current_sort_column == index do %>
                            <i class={"fas #{if @sort_direction == :asc, do: "fa-sort-up", else: "fa-sort-down"} text-blue-500"}></i>
                          <% else %>
                            <i class="fas fa-sort text-gray-300 group-hover:text-gray-500"></i>
                          <% end %>
                        </div>
                      </div>
                      <input
                        type="text"
                        placeholder="Filter..."
                        phx-keyup="column-search"
                        phx-debounce="300"
                        phx-value-column={index}
                        phx-target={@myself}
                        value={Map.get(@column_search, index, "")}
                        class="mt-1 text-xs p-1 border border-gray-200 rounded w-full"
                      />
                    </div>
                  </th>
                <% end %>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <%= if length(@filtered_rows) > 0 do %>
                <% start_index = (@page - 1) * @rows_per_page %>
                <% displayed_rows = Enum.slice(@filtered_rows, start_index, @rows_per_page) %>

                <%= for row <- displayed_rows do %>
                  <tr class="hover:bg-gray-50">
                    <%= for {cell, _cell_index} <- Enum.with_index(row) do %>
                      <td class="px-6 py-3 whitespace-nowrap text-sm text-gray-500">
                        <%= cell %>
                      </td>
                    <% end %>
                  </tr>
                <% end %>
              <% else %>
                <tr>
                  <td class="px-6 py-8 text-center text-sm text-gray-500" colspan={length(@headers)}>
                    <div class="flex flex-col items-center justify-center">
                      <i class="fas fa-search text-gray-300 text-3xl mb-2"></i>
                      <p>No matching records found</p>
                      <%= if @search_term != "" || map_size(@column_search) > 0 do %>
                        <button
                          phx-click="clear-filters"
                          phx-target={@myself}
                          class="mt-2 text-blue-600 hover:text-blue-800 text-sm"
                        >
                          Clear all filters
                        </button>
                      <% end %>
                    </div>
                  </td>
                </tr>
              <% end %>
            </tbody>
          </table>
        <% else %>
          <div class="p-8 text-center text-gray-500">
            <i class="fas fa-exclamation-triangle text-yellow-500 text-2xl mb-2"></i>
            <p>Unable to parse the CSV data. Please check the file format.</p>
          </div>
        <% end %>
      </div>

      <!-- Pagination controls -->
      <%= if length(@filtered_rows) > 0 && @total_pages > 1 do %>
        <div class="border-t border-gray-200 px-4 py-3 flex items-center justify-between">
          <div class="text-sm text-gray-700 hidden sm:block">
            Showing <span class="font-medium"><%= (@page - 1) * @rows_per_page + 1 %></span> to
            <span class="font-medium"><%= min(@page * @rows_per_page, length(@filtered_rows)) %></span> of
            <span class="font-medium"><%= length(@filtered_rows) %></span> results
          </div>
          <div class="flex items-center space-x-2">
            <button
              phx-click="change-page"
              phx-value-page="1"
              phx-target={@myself}
              disabled={@page == 1}
              class={"relative inline-flex items-center px-2 py-2 rounded-md border #{if @page == 1, do: "text-gray-300 border-gray-200", else: "text-gray-700 border-gray-300 hover:bg-gray-50"}"}>
              <i class="fas fa-angle-double-left text-xs"></i>
            </button>
            <button
              phx-click="change-page"
              phx-value-page={@page - 1}
              phx-target={@myself}
              disabled={@page == 1}
              class={"relative inline-flex items-center px-2 py-2 rounded-md border #{if @page == 1, do: "text-gray-300 border-gray-200", else: "text-gray-700 border-gray-300 hover:bg-gray-50"}"}>
              <i class="fas fa-angle-left text-xs"></i>
            </button>

            <%= for page_num <- max(1, @page - 2)..min(@total_pages, @page + 2) do %>
              <button
                phx-click="change-page"
                phx-value-page={page_num}
                phx-target={@myself}
                class={"relative inline-flex items-center px-4 py-2 border rounded-md #{if @page == page_num, do: "bg-blue-50 text-blue-600 border-blue-500", else: "text-gray-700 border-gray-300 hover:bg-gray-50"}"}>
                <%= page_num %>
              </button>
            <% end %>

            <button
              phx-click="change-page"
              phx-value-page={@page + 1}
              phx-target={@myself}
              disabled={@page == @total_pages}
              class={"relative inline-flex items-center px-2 py-2 rounded-md border #{if @page == @total_pages, do: "text-gray-300 border-gray-200", else: "text-gray-700 border-gray-300 hover:bg-gray-50"}"}>
              <i class="fas fa-angle-right text-xs"></i>
            </button>
            <button
              phx-click="change-page"
              phx-value-page={@total_pages}
              phx-target={@myself}
              disabled={@page == @total_pages}
              class={"relative inline-flex items-center px-2 py-2 rounded-md border #{if @page == @total_pages, do: "text-gray-300 border-gray-200", else: "text-gray-700 border-gray-300 hover:bg-gray-50"}"}>
              <i class="fas fa-angle-double-right text-xs"></i>
            </button>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  # Helper functions
  defp parse_csv(csv_string) do
    rows =
      csv_string
      |> String.split("\n", trim: true)
      |> Enum.map(fn line ->
        line
        |> String.split(",", trim: true)
        |> Enum.map(&String.trim/1)
      end)
      |> Enum.filter(fn row -> length(row) > 0 end)

    headers = List.first(rows) || []
    data_rows = if length(rows) > 1, do: tl(rows), else: []

    {headers, data_rows}
  end

  defp apply_column_filters(rows, column_search) do
    if map_size(column_search) == 0 do
      rows
    else
      rows
      |> Enum.filter(fn row ->
        Enum.all?(column_search, fn {column_index, search_term} ->
          cell = if length(row) > column_index, do: Enum.at(row, column_index), else: ""
          cell && String.downcase(cell) =~ String.downcase(search_term)
        end)
      end)
    end
  end
end
