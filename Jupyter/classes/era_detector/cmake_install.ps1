# Script to Download and Install Visual Studio 2022 Build Tools for C++ Development
# and then Verify CMake installation.

# --- Configuration ---
# Official Microsoft link for the latest Visual Studio 2022 Build Tools bootstrapper
# This link is commonly used and cited (e.g., in saikyun/install-cl GitHub repo)
$vsBuildToolsBootstrapperUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
$vsBootstrapperFileName = "vs_buildtools_bootstrapper.exe"
$vsBootstrapperPath = Join-Path -Path $env:TEMP -ChildPath $vsBootstrapperFileName

# Workload and components for C++ development, including CMake
# Microsoft.VisualStudio.Workload.VCTools is the main workload for C++ build tools.
# It generally includes CMake. --includeRecommended helps get common associated tools.
$vsInstallArguments = @(
    "--quiet", 
    "--wait", 
    "--norestart", 
    "--nocache",
    "--add", "Microsoft.VisualStudio.Workload.VCTools", # Core C++ Desktop development tools
    # "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project", # Ensures CMake is included if not default
    "--includeRecommended" 
)

# --- Functions ---
Function Test-CMakeInstallation {
    Write-Host ""
    Write-Host "--- Verifying CMake Installation ---" -ForegroundColor Cyan
    try {
        $cmakeVersionOutput = cmake --version
        if ($LASTEXITCODE -eq 0) {
            Write-Host "CMake appears to be installed and in PATH." -ForegroundColor Green
            Write-Host $cmakeVersionOutput
            Write-Host "If this is the first time after installation, and you didn't restart the PC,"
            Write-Host "sometimes a terminal restart is still needed for all environment variables to be fully active."
            return $true
        } else {
            Write-Host "CMake --version command executed but returned a non-zero exit code." -ForegroundColor Yellow
            Write-Host "Output: $cmakeVersionOutput"
            Write-Host "CMake might be installed but not functioning correctly, or PATH needs update (try restarting terminal/PC)." -ForegroundColor Yellow
            return $false
        }
    }
    catch {
        Write-Host "CMake command not found or failed to execute." -ForegroundColor Red
        Write-Host "This indicates CMake is likely not installed or not in your system PATH." -ForegroundColor Red
        Write-Host "Please ensure the Visual Studio Build Tools installation completed successfully,"
        Write-Host "and try restarting your PowerShell terminal or your computer." -ForegroundColor Red
        Write-Host "Detailed error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# --- Main Script ---
Write-Host "Starting Visual Studio 2022 Build Tools C++ installation..."

# 1. Download the VS Build Tools Bootstrapper
Write-Host ""
Write-Host "Step 1: Downloading Visual Studio Build Tools bootstrapper..." -ForegroundColor Cyan
Write-Host "URL: $vsBuildToolsBootstrapperUrl"
Write-Host "Output Path: $vsBootstrapperPath"
try {
    Invoke-WebRequest -Uri $vsBuildToolsBootstrapperUrl -OutFile $vsBootstrapperPath -UseBasicParsing
    If (Test-Path $vsBootstrapperPath) {
        Write-Host "Bootstrapper downloaded successfully." -ForegroundColor Green
    } Else {
        Write-Host "Failed to download the bootstrapper. Please check the URL and your internet connection." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error during download: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 2. Run the Bootstrapper to Install C++ Build Tools
Write-Host ""
Write-Host "Step 2: Installing Visual Studio Build Tools..." -ForegroundColor Cyan
Write-Host "This will take some time and run silently. Please be patient."
Write-Host "Arguments: $($vsInstallArguments -join ' ')"
try {
    $process = Start-Process -FilePath $vsBootstrapperPath -ArgumentList $vsInstallArguments -Wait -PassThru
    if ($process.ExitCode -eq 0) {
        Write-Host "Visual Studio Build Tools installation process completed. Exit Code: $($process.ExitCode)" -ForegroundColor Green
        Write-Host "An exit code of 0 usually means success, but some installers might use other codes (e.g., 3010 for success with reboot required)."
    } elseif ($process.ExitCode -eq 3010) {
        Write-Host "Visual Studio Build Tools installation process completed with Exit Code: 3010." -ForegroundColor Yellow
        Write-Host "This typically means success, but a system RESTART IS REQUIRED for changes to take full effect." -ForegroundColor Yellow
    } else {
        Write-Host "Visual Studio Build Tools installation process completed with Exit Code: $($process.ExitCode)." -ForegroundColor Yellow
        Write-Host "This might indicate an issue. Check logs in %TEMP% folder (search for files like dd_bootstrapper_*.log or vslogs.zip)." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Error running the Visual Studio Build Tools installer: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure you are running PowerShell as Administrator if you encounter permission issues." -ForegroundColor Yellow
    # Clean up installer on error too
    if (Test-Path $vsBootstrapperPath) {
        Remove-Item $vsBootstrapperPath -Force
        Write-Host "Cleaned up downloaded bootstrapper."
    }
    exit 1
}

# 3. Clean up the downloaded installer
Write-Host ""
Write-Host "Step 3: Cleaning up downloaded bootstrapper..." -ForegroundColor Cyan
if (Test-Path $vsBootstrapperPath) {
    try {
        Remove-Item $vsBootstrapperPath -Force
        Write-Host "Bootstrapper file removed successfully." -ForegroundColor Green
    }
    catch {
        Write-Host "Could not remove bootstrapper file: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Bootstrapper file not found, no cleanup needed."
}

# 4. Instructions for Verification
Write-Host ""
Write-Host "--- IMPORTANT NEXT STEPS ---" -ForegroundColor Yellow
Write-Host "1. It is STRONGLY RECOMMENDED to RESTART your computer now to ensure all PATH"
Write-Host "   and environment variable changes are applied correctly."
Write-Host "2. After restarting (or if you choose not to, at least close and REOPEN this PowerShell terminal),"
Write-Host "   you can verify the CMake installation by running the following command in the NEW terminal:"
Write-Host ""
Write-Host "   Test-CMakeInstallation" -ForegroundColor Magenta
Write-Host ""
Write-Host "You can also just type 'cmake --version' in the new terminal."
Write-Host "If CMake is found, the installation was likely successful."
Write-Host "Script finished."

# To automatically run the CMake test after this script, you could uncomment the line below.
# However, it's best to run it in a *new* terminal session after the install.
# Read-Host -Prompt "Press Enter to attempt CMake verification in this session (new terminal recommended)"
# Test-CMakeInstallation