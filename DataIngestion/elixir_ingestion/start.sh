#!/bin/sh
# Startup script to run both the data processor and web service

# Start the Phoenix web service in the background
echo "Starting Phoenix web service..."
cd /app/web_service
elixir --sname web -S mix phx.server &

# Wait for the web service to start
echo "Waiting for web service to start..."
sleep 5

# Display usage information
echo ""
echo "Data Processor is ready!"
echo "To process a file, use the following command in the container:"
echo "  elixir process_data_file.exs /path/to/data/file"
echo ""
echo "Web UI is available at: http://localhost:4000"
echo ""

# Keep the container running
echo "Container is now running. Press Ctrl+C to stop."
# Use tail -f /dev/null to keep the container running
exec tail -f /dev/null
