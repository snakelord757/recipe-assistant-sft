$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$Host.UI.RawUI.WindowTitle = 'Recipe SFT - live progress (UTF-8)'
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'results\console.log') -Encoding UTF8 -Tail 15 -Wait
