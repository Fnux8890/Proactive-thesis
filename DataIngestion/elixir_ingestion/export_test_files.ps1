# Script to export all test files to a single text file
$outputFile = Join-Path -Path (Get-Location) -ChildPath "all_test_files.txt"

# Clear the output file if it exists
if (Test-Path $outputFile) {
    Clear-Content $outputFile
}

# Get all files in the test directory recursively
$testFiles = Get-ChildItem -Path ".\pipeline\test" -Recurse -File | Where-Object { $_.Extension -eq ".exs" }

# Process each file and append to the output file
foreach ($file in $testFiles) {
    # Add a header with file path and separator
    Add-Content -Path $outputFile -Value "===================="
    Add-Content -Path $outputFile -Value "File: $($file.FullName)"
    Add-Content -Path $outputFile -Value "===================="
    
    # Append file content
    Get-Content -Path $file.FullName | Add-Content -Path $outputFile
    
    # Add a blank line between files
    Add-Content -Path $outputFile -Value "`n`n"
}

Write-Host "All test files have been exported to $outputFile" 