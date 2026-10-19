# Archive old selfplay match folders to E: as one .tar.zst per tag, VERIFY, then remove the bulky per-game files.
# 2026-10-18 (owner-approved plan, memory archive-old-selfplay-games). Nothing is lost: the archive is a full snapshot of
# the folder; the small top-level files (summary.csv, tournament.json/.pgn, *.csv, *.txt) STAY in place so tools keep working.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File selfplay\archive_games.ps1 -Before 2026-10-05 [-Limit N] [-NoDelete]
#
# Per tag:  tar (bsdtar --zstd, level 10) -> list archive: file count + byte sum must equal the disk walk (this also
# decompresses the whole stream, so a corrupt archive fails) -> extract one random file and compare SHA-256 with the
# original -> record archive SHA-256 in the manifest -> only then delete subfolders + top-level game_* files.
# Restore one tag:  tar.exe -xf E:\chess_data\archive\games\<tag>.tar.zst -C <games dir>
# Idempotent: tags already "deleted" in the manifest are skipped. Runs at Idle priority (child tar inherits it).
param(
    [string]$Before = "2026-10-05",
    [int]$Limit = 0,
    [string[]]$Tags = @(),     # explicit tags (skips the full scan; still must be older than -Before)
    [switch]$NoDelete
)
$ErrorActionPreference = "Stop"
[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = "Idle"

$G   = "C:\Users\Kumodth\OneDrive\Desktop\Programming\Chess Engine\Chess-Engine\NN Engine\selfplay\games"
$A   = "E:\chess_data\archive\games"
$V   = "E:\chess_data\archive\_verify"
$MAN = "E:\chess_data\archive\manifest.csv"
$LOG = "E:\chess_data\archive\archive_run.log"
New-Item -ItemType Directory -Force $A, $V | Out-Null
if (-not (Test-Path $MAN)) {
    "tag,newest,files,bytes,archive_bytes,sha256,spot_file,status,when" | Out-File $MAN -Encoding ascii
}
function Log($m) { $s = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; $s | Out-File $LOG -Append -Encoding utf8; $s }

$done = @{}
foreach ($r in (Import-Csv $MAN)) { if ($r.status -eq "deleted" -or ($NoDelete -and $r.status -eq "verified")) { $done[$r.tag] = 1 } }

# Candidate set: every match folder whose NEWEST file is older than -Before.
$cands = @()
$Tags = @($Tags | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # -File passes "a,b,c" as one string
$dirs = if ($Tags.Count) { $Tags | ForEach-Object { Join-Path $G $_ } } else { [System.IO.Directory]::EnumerateDirectories($G) }
foreach ($d in $dirs) {
    $di = New-Object System.IO.DirectoryInfo $d
    if ($done.ContainsKey($di.Name)) { continue }
    $newest = [datetime]::MinValue; $any = $false
    foreach ($f in $di.EnumerateFiles("*", [System.IO.SearchOption]::AllDirectories)) {
        $any = $true; if ($f.LastWriteTime -gt $newest) { $newest = $f.LastWriteTime }
    }
    if ($any -and $newest.ToString("yyyy-MM-dd") -lt $Before) { $cands += ,@($di, $newest) }
}
$cands = $cands | Sort-Object { $_[1] }
if ($Limit -gt 0) { $cands = $cands | Select-Object -First $Limit }
Log ("start: {0} tags before {1} (NoDelete={2})" -f @($cands).Count, $Before, [bool]$NoDelete)

$ok = 0; $fail = 0
foreach ($c in $cands) {
    $di = $c[0]; $tag = $di.Name; $phase = "archive"
    $files = @($di.EnumerateFiles("*", [System.IO.SearchOption]::AllDirectories))
    $nf = $files.Count; $nb = [long]($files | Measure-Object Length -Sum).Sum
    $arc = Join-Path $A "$tag.tar.zst"
    try {
        if (Test-Path $arc) { Remove-Item $arc -Force }   # a leftover from an interrupted run is never trusted
        & tar.exe --zstd --options "zstd:compression-level=10,zstd:threads=2" -cf $arc -C $G $tag
        if ($LASTEXITCODE -ne 0) { throw "tar create exit $LASTEXITCODE" }

        # Verify 1: full listing (decompresses everything) -> regular-file count and byte sum equal the disk walk.
        $lf = 0; $lb = [long]0
        foreach ($line in (& tar.exe -tvf $arc)) {
            if ($line.StartsWith("-")) { $lf++; $lb += [long](($line -split "\s+")[4]) }
        }
        if ($LASTEXITCODE -ne 0) { throw "tar list exit $LASTEXITCODE" }
        if ($lf -ne $nf -or $lb -ne $nb) { throw "listing mismatch: archive $lf files/$lb B vs disk $nf/$nb" }

        # Verify 2: extract one random file, compare SHA-256 with the original.
        $pick = $files[(Get-Random -Maximum $nf)]
        $rel  = $pick.FullName.Substring($G.Length + 1).Replace("\", "/")
        $vd = Join-Path $V $tag; if (Test-Path $vd) { Remove-Item $vd -Recurse -Force }
        New-Item -ItemType Directory -Force $vd | Out-Null
        & tar.exe -xf $arc -C $vd $rel
        if ($LASTEXITCODE -ne 0) { throw "spot extract exit $LASTEXITCODE" }
        $h1 = (Get-FileHash $pick.FullName -Algorithm SHA256).Hash
        $h2 = (Get-FileHash (Join-Path $vd $rel) -Algorithm SHA256).Hash
        Remove-Item $vd -Recurse -Force
        if ($h1 -ne $h2) { throw "spot hash mismatch on $rel" }

        $ab = (Get-Item $arc).Length
        $ah = (Get-FileHash $arc -Algorithm SHA256).Hash
        $status = "verified"
        if (-not $NoDelete) {
            $phase = "delete"
            # Remove only the bulky per-game data: every subfolder + top-level game_* files. Summaries stay.
            foreach ($s in $di.GetDirectories()) { Remove-Item $s.FullName -Recurse -Force }
            foreach ($f in $di.GetFiles("game_*")) { Remove-Item $f.FullName -Force }
            $status = "deleted"
        }
        ('"{0}",{1},{2},{3},{4},{5},"{6}",{7},{8}' -f $tag, $c[1].ToString("yyyy-MM-dd"), $nf, $nb, $ab, $ah, $rel, $status,
            (Get-Date -Format "yyyy-MM-dd HH:mm")) | Out-File $MAN -Append -Encoding ascii
        $ok++
        Log ("ok   {0}  {1} files  {2:N1} MB -> {3:N1} MB  [{4}]  ({5}/{6})" -f $tag, $nf, ($nb/1MB), ($ab/1MB), $status, ($ok+$fail), @($cands).Count)
    } catch {
        $fail++
        $note = if ($phase -eq "delete") { "VERIFIED archive kept; delete was PARTIAL - rerun or restore from archive" } else { "originals untouched" }
        Log ("FAIL {0} [{1}]: {2}  ({3})" -f $tag, $phase, $_.Exception.Message, $note)
    }
}
Log ("finished: ok={0} fail={1}" -f $ok, $fail)
