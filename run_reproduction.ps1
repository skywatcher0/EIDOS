$ErrorActionPreference = "Stop"

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath ".venv")) {
        py -3.12 -m venv .venv
    }

    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    & .\.venv\Scripts\python.exe reproduce.py @args
}
finally {
    Pop-Location
}
