$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$DistributionLine = wsl.exe --list --quiet | Select-String "Ubuntu" | Select-Object -First 1
if (-not $DistributionLine) {
    Write-Error "Ubuntu is not installed in WSL. Run install.ps1 first."
    exit 1
}
$Distribution = ($DistributionLine.ToString().Trim() -replace "^\*\s*", "")
$LinuxRoot = (& wsl.exe -d $Distribution -- bash -lc "wslpath -a '$Root'").Trim()
& wsl.exe -d $Distribution -- bash -lc "cd '$LinuxRoot' && bash ./run.sh"
