$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RuntimeDir = Join-Path $ScriptDir "runtime"

# Add runtime directory to PATH so DLLs can be found
$Env:PATH = "$RuntimeDir;$Env:PATH"
Write-Host "Runtime Environment Configured: $RuntimeDir"

# Check if components exist
if (-not (Test-Path "$ScriptDir\models")) {
    Write-Warning "Models directory not found! You may need to run 'python assets/download_models.py'."
}

if (-not (Test-Path "$RuntimeDir\onnxruntime.dll")) {
    Write-Warning "onnxruntime.dll not found in runtime! You may need to run 'assets/download_dlls.ps1'."
}

# Run the example
# We use Invoke-Expression or direct call. Direct call is safer for args.
Write-Host "Starting Qwen3-TTS..."
cargo run --example qwen3-tts --release -- $args
