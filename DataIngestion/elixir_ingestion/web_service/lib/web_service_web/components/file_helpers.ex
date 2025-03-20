defmodule WebServiceWeb.Components.FileHelpers do
  @moduledoc """
  Helper functions for file display and visualization in the FileWatcher.
  """

  @doc """
  Returns the appropriate FontAwesome icon class for a given file type.
  """
  @spec file_icon_class(atom) :: String.t()
  def file_icon_class(:csv), do: "fas fa-table text-green-600"
  def file_icon_class(:json), do: "fas fa-code text-yellow-600"
  def file_icon_class(:excel), do: "fas fa-file-excel text-green-700"
  def file_icon_class(:text), do: "fas fa-file-alt text-blue-600"
  def file_icon_class(:xml), do: "fas fa-file-code text-orange-600"
  def file_icon_class(_), do: "fas fa-file text-gray-600"

  @doc """
  Returns the appropriate color class for a file type visualization.
  """
  @spec file_type_color(atom) :: String.t()
  def file_type_color(:csv), do: "bg-green-500"
  def file_type_color(:json), do: "bg-yellow-500"
  def file_type_color(:excel), do: "bg-green-700"
  def file_type_color(:text), do: "bg-blue-500"
  def file_type_color(:xml), do: "bg-orange-500"
  def file_type_color(_), do: "bg-gray-500"

  @doc """
  Format file size into human-readable format.
  """
  @spec format_file_size(integer) :: String.t()
  def format_file_size(size) when size < 1024, do: "#{size} B"
  def format_file_size(size) when size < 1024 * 1024, do: "#{Float.round(size / 1024, 2)} KB"
  def format_file_size(size) when size < 1024 * 1024 * 1024, do: "#{Float.round(size / (1024 * 1024), 2)} MB"
  def format_file_size(size), do: "#{Float.round(size / (1024 * 1024 * 1024), 2)} GB"
end
