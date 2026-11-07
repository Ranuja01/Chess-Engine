# Machine health check for the engine box: free disk space, whether OneDrive is behaving, and optionally the size of
# selfplay\games. Prints one line per check, "note:" lines for things worth a look and "ALERT:" lines for faults.
# The exit code is the number of alerts.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File selfplay\machine_health.ps1            # quick (~10 s)
#   powershell -NoProfile -ExecutionPolicy Bypass -File selfplay\machine_health.ps1 -Games     # also sizes selfplay\games
#
# -Games walks every file under selfplay\games, which costs disk and CPU: never run it while a queue times anything.
#
# OneDrive background: its Microsoft Lists sync engine ("Nucleus", process OneDrive.Sync.Service.exe) once wrote ~1 MB
# log files into logs\ListSync\Local without ever rotating them, about 110 GB on C: in five weeks, and used about one core
# nonstop. The machine policy DisableNucleusSync = 1 (HKLM\SOFTWARE\Policies\Microsoft\OneDrive) turns Lists sync off.
# File sync (OneDrive.exe) also logs heavily, but it rotates and compresses its logs, so that folder stays small.
param([switch]$Games)

$alerts = 0
function Alert([string]$m) { "ALERT: $m"; $script:alerts++ }

# free disk space (C: holds the WSL disk image; a full C: fails WSL writes)
$cFree = (Get-PSDrive C).Free / 1GB
"C: free {0:N1} GB; E: free {1:N1} GB" -f $cFree, ((Get-PSDrive E -ErrorAction SilentlyContinue).Free / 1GB)
if ($cFree -lt 12) { Alert "C: under 12 GB free" } elseif ($cFree -lt 40) { "note: C: under 40 GB free, see the selfplay\games routine" }

# OneDrive processes, average CPU over 10 s (100 = one full core)
$names = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in 'OneDrive', 'OneDrive.Sync.Service' } |
    Select-Object -ExpandProperty ProcessName -Unique)
$cpu = @{}
if ($names.Count -gt 0) {
    $paths = $names | ForEach-Object { "\Process($_)\% Processor Time" }
    $samples = (Get-Counter $paths -SampleInterval 1 -MaxSamples 10 -ErrorAction SilentlyContinue).CounterSamples
    foreach ($n in $names) {
        $v = $samples | Where-Object { $_.InstanceName -eq $n.ToLower() } | Measure-Object CookedValue -Average
        $cpu[$n] = [int]$v.Average
    }
}
if ($names -contains 'OneDrive') {
    "OneDrive file sync running, CPU {0}% of a core" -f $cpu['OneDrive']
    if ($cpu['OneDrive'] -gt 50) { "note: file sync is busy; normal for the first hours after a restart or a big file change, not at idle" }
} else {
    Alert "OneDrive file sync (OneDrive.exe) is not running, so nothing is being backed up to the cloud"
}
if ($names -contains 'OneDrive.Sync.Service') {
    "OneDrive.Sync.Service running, CPU {0}% of a core" -f $cpu['OneDrive.Sync.Service']
    if ($cpu['OneDrive.Sync.Service'] -gt 20) { Alert "OneDrive.Sync.Service is busy; it hosts the Lists sync engine that ran away before" }
}

# the Lists sync policy and its log folder
$pol = (Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\OneDrive' -ErrorAction SilentlyContinue).DisableNucleusSync
"policy DisableNucleusSync = $(if ($null -eq $pol) { '(not set)' } else { $pol })"
if ($pol -ne 1) { "note: Lists sync is allowed; it is the engine whose logs filled C: before" }
$logs = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs'
$ls = @(Get-ChildItem (Join-Path $logs 'ListSync\Local') -File -Filter *.odl -ErrorAction SilentlyContinue)
"ListSync\Local: {0} .odl files, {1:N1} MB" -f $ls.Count, (($ls | Measure-Object Length -Sum).Sum / 1MB)
if ($ls.Count -gt 100) { Alert "the Lists sync log loop is back (ListSync\Local keeps growing)" }
$all = @(Get-ChildItem $logs -File -Recurse -ErrorAction SilentlyContinue)
$allGB = ($all | Measure-Object Length -Sum).Sum / 1GB
"OneDrive logs total: {0} files, {1:N2} GB nominal" -f $all.Count, $allGB
if ($allGB -gt 2) { Alert "OneDrive's logs are over 2 GB; find the subfolder that keeps growing" }

# selfplay\games (archive tags older than ~2 weeks with archive_games.ps1 once it passes ~10 GB)
if ($Games) {
    $g = Join-Path $PSScriptRoot 'games'
    $f = @(Get-ChildItem $g -File -Recurse -ErrorAction SilentlyContinue)
    $gGB = ($f | Measure-Object Length -Sum).Sum / 1GB
    "selfplay\games: {0:N2} GB, {1} files, {2} tags" -f $gGB, $f.Count, @(Get-ChildItem $g -Directory -ErrorAction SilentlyContinue).Count
    if ($gGB -gt 10) { Alert "selfplay\games is over 10 GB: archive old tags with selfplay\archive_games.ps1 -Before <date>" }
}

exit $alerts
