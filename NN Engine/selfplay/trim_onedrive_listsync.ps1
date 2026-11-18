# Deletes OneDrive's Lists-sync diagnostic logs (logs\ListSync\Local\*.odl) older than -Hours (default 12).
# OneDrive.Sync.Service (v26.195) writes ~740 of these ~1 MB files an hour in a loop and never removes them; the
# DisableNucleusSync policy does not stop it. Unchecked they filled ~100 GB of C: by 11-07 and ~38 GB by 11-18. Run every
# 2 hours by the scheduled task "Trim OneDrive ListSync logs" (registered 2026-11-18, no admin), so the folder holds about
# half a day of them (~3 GB on disk). Remove the task to stop: Unregister-ScheduledTask -TaskName 'Trim OneDrive ListSync logs'
# Only *.odl in that one folder are touched; Microsoft.LocalContent.db (the live Lists store, in ListSync\Common) never is.
param([int]$Hours = 12)
$dir = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\ListSync\Local'
$log = Join-Path $env:LOCALAPPDATA 'trim_onedrive_listsync.log'
if (-not (Test-Path -LiteralPath $dir)) { return }
$cut = (Get-Date).AddHours(-$Hours)
$n = 0; $kept = 0; $fail = 0
foreach ($f in [System.IO.Directory]::EnumerateFiles($dir, '*.odl')) {
	try {
		if ([System.IO.File]::GetLastWriteTime($f) -lt $cut) { [System.IO.File]::Delete($f); $n++ } else { $kept++ }
	} catch { $fail++ }
}
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Add-Content -LiteralPath $log -Value ("{0} deleted {1}, kept {2}, failed {3}, C: free {4} GB" -f (Get-Date -Format s), $n, $kept, $fail, $free)
