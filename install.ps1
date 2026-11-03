$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path

function Fail($Message) {
    Write-Error $Message
    exit 1
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Host "Windows installation uses WSL because upstream glafic is Unix-oriented."
    Write-Host "Installing Ubuntu through WSL. Windows may require a restart; rerun install.ps1 after the restart."
    wsl.exe --install --no-launch -d Ubuntu
    if ($LASTEXITCODE -ne 0) {
        Fail "WSL installation failed. Install WSL and Ubuntu, then rerun this script."
    }
    Write-Host "WSL was installed. Restart Windows if requested, then rerun install.ps1."
    exit 0
}

$DistributionLine = wsl.exe --list --quiet | Select-String "Ubuntu" | Select-Object -First 1
if (-not $DistributionLine) {
    Fail "Ubuntu is not installed in WSL. Run: wsl --install -d Ubuntu"
}
$Distribution = ($DistributionLine.ToString().Trim() -replace "^\*\s*", "")

$LinuxRoot = (& wsl.exe -d $Distribution -- bash -lc "wslpath -a '$Root'").Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxRoot) {
    Fail "Could not map the project directory into WSL."
}

Write-Host "Installing the application inside WSL Ubuntu..."
& wsl.exe -d $Distribution -- bash -lc "cd '$LinuxRoot' && bash ./install.sh"
if ($LASTEXITCODE -ne 0) {
    Fail "The WSL installation failed."
}

Write-Host ""
Write-Host "Installation complete. Start the application with:"
Write-Host "  wsl -d $Distribution -- bash -lc `"cd '$LinuxRoot' && bash ./run.sh`""
