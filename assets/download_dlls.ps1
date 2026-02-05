<#
.SYNOPSIS
Downloads required DLLs for Qwen3-TTS (GPU Support).

.DESCRIPTION
Downloads ONNX Runtime 1.19.2 with CUDA support and extracts necessary DLLs to the runtime folder.
#>

$ORT_VERSION = "1.23.2"
$ORT_URL_CPU = "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-win-x64-$ORT_VERSION.zip"
$ZIP_NAME = "onnxruntime-cpu.zip"

# Determine Runtime Directory (Root/runtime)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RuntimeDir = Join-Path $ScriptDir "..\runtime"

if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
    Write-Host "Created runtime directory: $RuntimeDir"
}

Write-Host "=========================================="
Write-Host "  Qwen3-TTS Dependency Downloader"
Write-Host "  Target: $RuntimeDir"
Write-Host "=========================================="

# 1. Download ONNX Runtime (CPU)
if (-not (Test-Path "$RuntimeDir\onnxruntime.dll")) {
    Write-Host "[1/2] Downloading ONNX Runtime (CPU) v$ORT_VERSION..."
    
    try {
        Invoke-WebRequest -Uri $ORT_URL_CPU -OutFile $ZIP_NAME -UseBasicParsing
    } catch {
        Write-Error "Failed to download ONNX Runtime. Check your internet connection."
        exit 1
    }

    Write-Host "[2/2] Extracting DLLs..."
    Expand-Archive -Path $ZIP_NAME -DestinationPath "tmp_extract" -Force

    # Files to copy
    $DllsToCopy = @(
        "onnxruntime.dll"
    )

    $SourcePath = "tmp_extract\onnxruntime-win-x64-$ORT_VERSION\lib"
    
    foreach ($dll in $DllsToCopy) {
        $Src = Join-Path $SourcePath $dll
        if (Test-Path $Src) {
            Copy-Item $Src -Destination $RuntimeDir -Force
            Write-Host "  + Installed $dll"
        } else {
            Write-Warning "  ! Could not find $dll in archive."
        }
    }

    # Cleanup
    Remove-Item "tmp_extract" -Recurse -Force
    Remove-Item $ZIP_NAME -Force
    Write-Host "ONNX Runtime installed successfully."
} else {
    Write-Host "ONNX Runtime DLLs already exist. Skipping."
}

Write-Host "`nWARNING: Llama.cpp and GGML DLLs are NOT automatically downloaded yet."
Write-Host "Please manually place the following files in '$RuntimeDir':"
Write-Host "  - llama.dll"
Write-Host "  - ggml.dll"
Write-Host "  - ggml-cuda.dll (if using CUDA)"
Write-Host "=========================================="
Write-Host "Done."
