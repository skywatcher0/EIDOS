$ErrorActionPreference = 'Stop'
& py -3.12 (Join-Path $PSScriptRoot 'bootstrap.py') @args
exit $LASTEXITCODE
