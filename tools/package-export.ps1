$ErrorActionPreference = "Stop"

$version = "0.1.0"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$root = Split-Path -Parent $PSScriptRoot
$distRoot = Join-Path $root "dist\\obs-face-emotion-filter"
$exportDir = Join-Path $root "export"
$packageName = "OBS-Face-Emotion-Filter-$version-portable-$timestamp"
$packageRoot = Join-Path $exportDir $packageName

if (-not (Test-Path $distRoot)) {
  throw "Dossier dist introuvable: $distRoot"
}

$pluginRoot = Join-Path $packageRoot "obs-face-emotion-filter"
$binDst = Join-Path $pluginRoot "bin\\64bit"
$dataDst = Join-Path $pluginRoot "data"

New-Item -ItemType Directory -Path $binDst -Force | Out-Null
New-Item -ItemType Directory -Path $dataDst -Force | Out-Null

Copy-Item (Join-Path $distRoot "bin\\64bit\\*") $binDst -Force
$pdb = Join-Path $binDst "obs-face-emotion-filter.pdb"
if (Test-Path $pdb) {
  Remove-Item $pdb -Force
}
Copy-Item (Join-Path $distRoot "data\\*") $dataDst -Recurse -Force

$installFile = Join-Path $packageRoot "INSTALL-FR.txt"
$installLines = @(
  "OBS Face Emotion Filter - Export portable"
  ""
  "Installation (Admin):"
  "1) Ferme OBS Studio."
  "2) Ouvre PowerShell en administrateur."
  "3) Lance le script: .\\install-admin.ps1"
  "4) Relance OBS."
  ""
  "Installation manuelle:"
  "- Copie obs-face-emotion-filter\\bin\\64bit\\* dans:"
  "  C:\\Program Files\\obs-studio\\obs-plugins\\64bit\\"
  "- Copie obs-face-emotion-filter\\data\\* dans:"
  "  C:\\Program Files\\obs-studio\\data\\obs-plugins\\obs-face-emotion-filter\\"
  ""
  "Desinstallation:"
  "- Supprimer les fichiers copies ci-dessus."
)
Set-Content -Path $installFile -Value $installLines -Encoding UTF8

$installScript = Join-Path $packageRoot "install-admin.ps1"
$scriptLines = @(
  '$ErrorActionPreference = "Stop"'
  '$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path'
  '$src = Join-Path $scriptRoot "obs-face-emotion-filter"'
  '$obs = "C:\Program Files\obs-studio"'
  ''
  'if (-not (Test-Path $src)) {'
  '  throw "Dossier source introuvable: $src"'
  '}'
  ''
  'Write-Host "Copie des DLL plugin..."'
  'Copy-Item (Join-Path $src "bin\64bit\*") (Join-Path $obs "obs-plugins\64bit\") -Force'
  ''
  'Write-Host "Copie des data plugin..."'
  '$dstData = Join-Path $obs "data\obs-plugins\obs-face-emotion-filter"'
  'New-Item -ItemType Directory -Path $dstData -Force | Out-Null'
  'Copy-Item (Join-Path $src "data\*") $dstData -Recurse -Force'
  ''
  'Write-Host "Installation terminee. Redemarre OBS Studio." -ForegroundColor Green'
)
Set-Content -Path $installScript -Value $scriptLines -Encoding UTF8

$zipPath = Join-Path $exportDir ($packageName + ".zip")
if (Test-Path $zipPath) {
  Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal

$sha = (Get-FileHash $zipPath -Algorithm SHA256).Hash
Write-Output "ZIP=$zipPath"
Write-Output "SHA256=$sha"
Write-Output "DIR=$packageRoot"
