defmodule IngestionServiceWeb.FallbackController do
  @moduledoc """
  Fallback controller for handling error responses.

  This module provides centralized error handling for all controllers,
  ensuring consistent error responses across the API.
  """

  use IngestionServiceWeb, :controller

  @doc """
  Renders a 404 not found response when a resource is not found.

  ## Parameters

  * `conn` - The connection struct
  * `{:error, :not_found}` - The error tuple

  ## Returns

  * JSON response with error details
  """
  def call(conn, {:error, :not_found}) do
    conn
    |> put_status(:not_found)
    |> put_view(json: IngestionServiceWeb.ErrorJSON)
    |> render(:"404")
  end

  @doc """
  Renders a 422 unprocessable entity response when there are validation errors.

  ## Parameters

  * `conn` - The connection struct
  * `{:error, %Ecto.Changeset{}}` - The error changeset

  ## Returns

  * JSON response with validation error details
  """
  def call(conn, {:error, %Ecto.Changeset{} = changeset}) do
    conn
    |> put_status(:unprocessable_entity)
    |> put_view(json: IngestionServiceWeb.ErrorJSON)
    |> render(:"422", changeset: changeset)
  end

  @doc """
  Renders a 400 bad request response when there are general errors.

  ## Parameters

  * `conn` - The connection struct
  * `{:error, reason}` - The error reason

  ## Returns

  * JSON response with error details
  """
  def call(conn, {:error, reason}) do
    conn
    |> put_status(:bad_request)
    |> put_view(json: IngestionServiceWeb.ErrorJSON)
    |> render(:"400", reason: reason)
  end
end
