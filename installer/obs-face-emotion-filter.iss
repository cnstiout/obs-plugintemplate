#define AppName "OBS Face Emotion Filter"
#define AppVersion "0.1.0"
#define PluginId "obs-face-emotion-filter"
#define DistRoot "..\\dist\\obs-face-emotion-filter"

[Setup]
AppId={{0B61D1F8-55A7-467F-A5CB-6E7648221A61}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=koka
DefaultDirName={autopf}\obs-studio
DisableDirPage=no
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=OBS-Face-Emotion-Filter-Setup-{#AppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Uninstallable=yes

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "startobs"; Description: "Lancer OBS Studio apres installation"; Flags: unchecked

[Files]
Source: "{#DistRoot}\bin\64bit\*"; DestDir: "{app}\obs-plugins\64bit"; Flags: ignoreversion
Source: "{#DistRoot}\data\*"; DestDir: "{app}\data\obs-plugins\{#PluginId}"; Flags: recursesubdirs createallsubdirs

[Run]
Filename: "{app}\bin\64bit\obs64.exe"; Description: "Lancer OBS Studio"; Flags: nowait postinstall skipifsilent; Tasks: startobs
