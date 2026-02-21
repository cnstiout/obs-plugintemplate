# OBS Face Emotion Filter

Plugin OBS natif C++ (Windows) qui ajoute un **filtre source webcam** pour:
- tracker jusqu'a 3 visages (rectangle),
- afficher une emotion en francais (`Joie`, `Tristesse`, `Colere`, `Peur`, `Surprise`, `Degout`, `Neutre`, `Incertain`),
- tourner en local/offline (aucune API cloud).

## Caracteristiques
- Cible OBS: `32.x` (buildspec verrouille `32.0.2`)
- Pipeline live: stream possible en `1080p60`, inference a `15 fps`
- Multi-visages: jusqu'a `3`
- Perte visage: masquage immediat
- Lissage emotions: `0.6s` (EMA)
- Seuil `Incertain`: `30%`
- Logs perf toutes les 5s (latence inference + fps inference + backlog queue)

## Prerequis (Windows)
- OBS Studio 32.x
- CMake >= 3.28
- Visual Studio Build Tools (generator configure dans `CMakePresets.json`)
- Windows SDK >= 10.0.20348.0
- vcpkg (pour `opencv4`)
- Inno Setup 6 (optionnel, pour l'installeur `.exe`)

## Build local
1. Configurer `VCPKG_ROOT` vers ton installation vcpkg.
2. Configurer CMake:

```powershell
cmake --preset windows-x64 -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
```

3. Compiler en Release:

```powershell
cmake --build --preset windows-x64 --config Release
```

4. Installer dans un dossier de distribution local:

```powershell
cmake --install build_x64 --config Release --prefix dist
```

Resultat attendu:
- Binaires plugin: `dist/obs-face-emotion-filter/bin/64bit/` (DLL plugin + DLL runtime OpenCV/protobuf)
- Donnees plugin/modeles: `dist/obs-face-emotion-filter/data/...`

## Installation manuelle dans OBS
Copier:
- `dist/obs-face-emotion-filter/bin/64bit/*` -> `C:\Program Files\obs-studio\obs-plugins\64bit\`
- `dist/obs-face-emotion-filter/data/` -> `C:\Program Files\obs-studio\data\obs-plugins\obs-face-emotion-filter\`

Puis redemarrer OBS.

## Creation de l'installeur Inno Setup
Depuis le dossier projet:

```powershell
iscc installer\obs-face-emotion-filter.iss
```

Le script attend les artefacts dans `dist/obs-face-emotion-filter`.

## Release macOS depuis le cloud (GitHub Actions)
Workflow disponible: `.github/workflows/release-macos-cloud.yml`

1. Push le projet sur GitHub.
2. Ouvre `Actions` -> `macOS Cloud Release`.
3. `Run workflow` avec un tag (ex: `v0.1.0`).
4. Recupere les fichiers dans:
- l'onglet `Artifacts` du run (`macos-release-assets`),
- ou la `Release` GitHub si `publish_release=true`.

## Utilisation dans OBS
1. Ajouter ta webcam dans une scene.
2. Ouvrir `Filtres` sur la webcam.
3. Ajouter `Face Emotion Filter`.
4. Regler:
- `Max Faces` (1..3),
- `Inference FPS` (defaut 15),
- `Inference Width` (defaut 640),
- `Confidence Threshold` (defaut 0.30),
- `Smoothing (seconds)` (defaut 0.6),
- options d'affichage rectangle/label/score.

## Modeles embarques
- `data/models/face_detection_yunet_2023mar.onnx`
- `data/models/emotion-ferplus-8.onnx`

## Limites connues
- Le filtre CPU supporte explicitement les formats d'image OBS BGRA/BGRX/RGBA/RGBX.
- Les performances dependent fortement du CPU.
- La qualite emotion peut varier selon eclairage, angle et resolution visage.

## Licence
Code source sous licence MIT.  
Dependances et modeles: voir `THIRD_PARTY_NOTICES.md`.
