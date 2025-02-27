defmodule IngestionServiceWeb.ErrorJSON do
  @moduledoc """
  JSON view for error responses.

  This module defines rendering functions for various HTTP error responses,
  ensuring consistent error message format across the API.
  """

  # If you want to customize a particular status code,
  # you may add your own clauses, such as:
  #
  # def render("500.json", _assigns) do
  #   %{errors: %{detail: "Internal Server Error"}}
  # end

  # By default, Phoenix returns the status message from
  # the template name. For example, "404.json" becomes
  # "Not Found".
  def render(template, _assigns) do
    %{errors: %{detail: Phoenix.Controller.status_message_from_template(template)}}
  end

  @doc """
  Renders a 400 bad request error.

  ## Parameters

  * `%{reason: reason}` - Map containing the error reason

  ## Returns

  * Map with error detail
  """
  def render("400.json", %{reason: reason}) do
    %{
      errors: %{
        detail: "Bad Request",
        reason: error_message(reason)
      }
    }
  end

  @doc """
  Renders a 422 unprocessable entity error with changeset errors.

  ## Parameters

  * `%{changeset: changeset}` - Map containing the changeset with errors

  ## Returns

  * Map with validation error details
  """
  def render("422.json", %{changeset: changeset}) do
    %{
      errors: %{
        detail: "Unprocessable Entity",
        errors: format_changeset_errors(changeset)
      }
    }
  end

  # Private helper functions

  defp error_message(reason) when is_atom(reason), do: Atom.to_string(reason)
  defp error_message(reason) when is_binary(reason), do: reason
  defp error_message(reason), do: inspect(reason)

  defp format_changeset_errors(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Regex.replace(~r"%{(\w+)}", msg, fn _, key ->
        opts |> Keyword.get(String.to_existing_atom(key), key) |> to_string()
      end)
    end)
  end
end
