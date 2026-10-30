$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
New-Item -ItemType Directory -Force -Path results | Out-Null
& .\.venv\Scripts\python.exe -u experiment.py all --epochs 8 2>&1 | Tee-Object -FilePath results\console.log
if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $LASTEXITCODE" }
