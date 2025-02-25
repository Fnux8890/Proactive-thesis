defmodule IngestionService.ErrorJSON do
  # If you want to customize a particular status code, you can add a clause to render/2
  # with that status_code. For example:
  #
  # def render("500.json", _assigns) do
  #   %{errors: %{detail: "Internal Server Error"}}
  # end

  # By default, Phoenix returns the status message from
  # the template name. For example, "404.json" becomes
  # "Not Found".
  def render(template, _assigns) do
    status_code = String.split(template, ".") |> hd() |> String.to_integer()
    status_message = get_status_message(status_code)
    
    %{
      status: :error,
      code: status_code,
      message: status_message
    }
  end
  
  # Map of standard HTTP status codes to their messages
  defp get_status_message(code) do
    %{
      400 => "Bad Request",
      401 => "Unauthorized",
      403 => "Forbidden",
      404 => "Not Found",
      405 => "Method Not Allowed",
      406 => "Not Acceptable",
      408 => "Request Timeout",
      409 => "Conflict",
      415 => "Unsupported Media Type",
      422 => "Unprocessable Entity",
      429 => "Too Many Requests",
      500 => "Internal Server Error",
      501 => "Not Implemented",
      502 => "Bad Gateway",
      503 => "Service Unavailable",
      504 => "Gateway Timeout"
    }
    |> Map.get(code, "Unknown Error")
  end
  
  # Convert Ecto validation errors into a standardized format
  def error(%{changeset: changeset}) do
    # Format the changeset errors
    errors = Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Regex.replace(~r"%{(\w+)}", msg, fn _, key ->
        opts |> Keyword.get(String.to_existing_atom(key), key) |> to_string()
      end)
    end)
    
    %{
      status: :error,
      errors: errors
    }
  end
  
  # Handle general errors
  def error(%{message: message}) do
    %{
      status: :error,
      message: message
    }
  end
  
  def error(message) when is_binary(message) do
    %{
      status: :error,
      message: message
    }
  end
end 