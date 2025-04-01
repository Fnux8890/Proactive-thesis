# Script to execute both export scripts
Write-Host "Starting to export all files..."

# Execute the lib files export script
Write-Host "Exporting lib files..."
& .\export_lib_files.ps1

# Execute the test files export script
Write-Host "Exporting test files..."
& .\export_test_files.ps1

Write-Host "All exports completed successfully!" 