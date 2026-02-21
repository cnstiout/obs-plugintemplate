$ErrorActionPreference = "Stop"

$version = "0.1.0"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$root = Split-Path -Parent $PSScriptRoot
$exportDir = Join-Path $root "export"
$packageName = "OBS-Face-Emotion-Filter-$version-macos-src-$timestamp"
$packageRoot = Join-Path $exportDir $packageName
$sourceRoot = Join-Path $packageRoot "obs-face-emotion-filter-src"

function Write-Utf8NoBom {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Content
  )

  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null

$itemsToCopy = @(
  "cmake"
  "build-aux"
  "src"
  "data"
  "installer"
  "CMakeLists.txt"
  "CMakePresets.json"
  "buildspec.json"
  "vcpkg.json"
  "LICENSE"
  "README.md"
  "THIRD_PARTY_NOTICES.md"
)

foreach ($item in $itemsToCopy) {
  $from = Join-Path $root $item
  if (-not (Test-Path $from)) {
    throw "Element manquant pour export macOS: $item"
  }
  Copy-Item $from (Join-Path $sourceRoot $item) -Recurse -Force
}

$macBuildScript = @(
  '#!/usr/bin/env bash'
  'set -euo pipefail'
  ''
  'ROOT="$(cd "$(dirname "$0")/obs-face-emotion-filter-src" && pwd)"'
  'cd "$ROOT"'
  ''
  'echo "[1/4] Configure macOS preset"'
  'cmake --preset macos'
  ''
  'echo "[2/4] Build plugin"'
  'cmake --build --preset macos --config RelWithDebInfo'
  ''
  'echo "[3/4] Install artifacts to dist_macos"'
  'cmake --install build_macos --config RelWithDebInfo --prefix dist_macos'
  ''
  'echo "[4/4] Done"'
  'echo "Output expected under: $ROOT/dist_macos"'
) -join "`n"
Write-Utf8NoBom -Path (Join-Path $packageRoot "build-macos.sh") -Content $macBuildScript

$macInstallUserScript = @(
  '#!/usr/bin/env bash'
  'set -euo pipefail'
  ''
  'ROOT="$(cd "$(dirname "$0")/obs-face-emotion-filter-src" && pwd)"'
  'DIST="$ROOT/dist_macos"'
  'PLUGIN_DIR="$HOME/Library/Application Support/obs-studio/plugins"'
  ''
  'if [[ ! -d "$DIST" ]]; then'
  '  echo "Missing $DIST. Run ./build-macos.sh first."'
  '  exit 1'
  'fi'
  ''
  'mkdir -p "$PLUGIN_DIR"'
  ''
  'PLUGIN_PATH="$(find "$DIST" -maxdepth 1 -type d -name "*.plugin" | head -n 1)"'
  'if [[ -z "${PLUGIN_PATH:-}" ]]; then'
  '  echo "No .plugin found in $DIST"'
  '  exit 1'
  'fi'
  ''
  'cp -R "$PLUGIN_PATH" "$PLUGIN_DIR/"'
  'echo "Installed: $PLUGIN_PATH -> $PLUGIN_DIR"'
  'echo "Restart OBS Studio."'
) -join "`n"
Write-Utf8NoBom -Path (Join-Path $packageRoot "install-macos-user.sh") -Content $macInstallUserScript

$macInstallGlobalScript = @(
  '#!/usr/bin/env bash'
  'set -euo pipefail'
  ''
  'ROOT="$(cd "$(dirname "$0")/obs-face-emotion-filter-src" && pwd)"'
  'DIST="$ROOT/dist_macos"'
  'PLUGIN_DIR="/Library/Application Support/obs-studio/plugins"'
  ''
  'if [[ ! -d "$DIST" ]]; then'
  '  echo "Missing $DIST. Run ./build-macos.sh first."'
  '  exit 1'
  'fi'
  ''
  'sudo mkdir -p "$PLUGIN_DIR"'
  ''
  'PLUGIN_PATH="$(find "$DIST" -maxdepth 1 -type d -name "*.plugin" | head -n 1)"'
  'if [[ -z "${PLUGIN_PATH:-}" ]]; then'
  '  echo "No .plugin found in $DIST"'
  '  exit 1'
  'fi'
  ''
  'sudo cp -R "$PLUGIN_PATH" "$PLUGIN_DIR/"'
  'echo "Installed globally: $PLUGIN_PATH -> $PLUGIN_DIR"'
  'echo "Restart OBS Studio."'
) -join "`n"
Write-Utf8NoBom -Path (Join-Path $packageRoot "install-macos-global.sh") -Content $macInstallGlobalScript

$readmeMac = @(
  'OBS Face Emotion Filter - Export macOS (source + scripts)'
  ''
  'Cet export est prepare depuis Windows et ne contient pas de binaire macOS precompile.'
  'Il est fait pour etre build sur un Mac.'
  ''
  'Prerequis sur Mac:'
  '- Xcode Command Line Tools'
  '- CMake >= 3.28'
  '- Dependances OpenCV disponibles pour CMake (OpenCV CONFIG)'
  ''
  'Build sur Mac:'
  '1) Dezipper ce package'
  '2) Ouvrir Terminal dans le dossier dezippe'
  '3) chmod +x build-macos.sh install-macos-user.sh install-macos-global.sh'
  '4) ./build-macos.sh'
  ''
  'Installation plugin:'
  '- User only: ./install-macos-user.sh'
  '- Global (tous users): ./install-macos-global.sh'
  ''
  'Chemins macOS OBS:'
  '- User: ~/Library/Application Support/obs-studio/plugins/'
  '- Global: /Library/Application Support/obs-studio/plugins/'
) -join "`r`n"
Write-Utf8NoBom -Path (Join-Path $packageRoot "INSTALL-MACOS.txt") -Content $readmeMac

$zipPath = Join-Path $exportDir ($packageName + ".zip")
if (Test-Path $zipPath) {
  Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal

$sha = (Get-FileHash $zipPath -Algorithm SHA256).Hash
Write-Output "ZIP=$zipPath"
Write-Output "SHA256=$sha"
Write-Output "DIR=$packageRoot"
