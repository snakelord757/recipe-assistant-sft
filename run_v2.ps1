$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
New-Item -ItemType Directory -Force -Path runs\v2 | Out-Null
& .\.venv\Scripts\python.exe -u experiment_v2.py 2>&1 | Tee-Object -FilePath runs\v2\console.log
if ($LASTEXITCODE -ne 0) { throw "Experiment V2 failed: $LASTEXITCODE" }
