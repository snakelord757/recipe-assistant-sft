$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding=$utf8
[Console]::OutputEncoding=$utf8
$OutputEncoding=$utf8
$Host.UI.RawUI.WindowTitle='Recipe SFT V2 - 4360 examples - UTF-8'
Write-Host 'V2: 4360 examples | batch=2 | 2180 batches/epoch | 273 updates/epoch'
Write-Host 'Order: V1 new-test baseline -> SFT -> V2 tests -> review'
Write-Host 'seen = examples processed in this epoch. mean = mean train loss.'
$log=Join-Path $PSScriptRoot 'runs\v2\console.log'
while (-not (Test-Path -LiteralPath $log)) { Start-Sleep -Seconds 1 }
Get-Content -LiteralPath $log -Encoding UTF8 -Tail 10 -Wait
