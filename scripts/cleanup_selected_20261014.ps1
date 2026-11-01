$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath('D:\SEG').TrimEnd('\')
$plan = Get-Content -LiteralPath "$workspaceRoot\reports\transition_20261014\cleanup_plan.json" -Raw | ConvertFrom-Json
if (-not $plan.archive_crc_verified) { throw 'Archive not verified' }
if ((Get-FileHash -LiteralPath $plan.archive -Algorithm SHA256).Hash.ToLower() -ne $plan.archive_sha256) { throw 'Archive hash mismatch' }
function Assert-WorkspacePath([string]$candidate) {
    $absolute = [IO.Path]::GetFullPath($candidate)
    if (-not $absolute.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
    if ($absolute.StartsWith($workspaceRoot + '\.git', [StringComparison]::OrdinalIgnoreCase) -or $absolute.StartsWith($workspaceRoot + '\.venv310', [StringComparison]::OrdinalIgnoreCase)) { throw "Protected: $absolute" }
    return $absolute
}
$deleted = @()
foreach ($entry in $plan.targets) {
    $target = Assert-WorkspacePath $entry.path
    if (Test-Path -LiteralPath $target) {
        $item = Get-Item -LiteralPath $target -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse target: $target" }
        if ($item.PSIsContainer) {
            $links = @(Get-ChildItem -LiteralPath $target -Force -Recurse -Attributes ReparsePoint)
            if ($links.Count -gt 0) { throw "Reparse children: $target" }
        }
        Remove-Item -LiteralPath $target -Force -Recurse
    }
    $deleted += $entry
}
# Historical scripts and documents are preserved in the verified ZIP.
$keepScripts = @('train_selected.py','fetch_selected_20261014.py','archive_retired_20261014.py','cleanup_selected_20261014.ps1','evaluate_features.py','update_status.py','verify_selected.py','finalize_selected.py')
foreach ($item in Get-ChildItem -LiteralPath "$workspaceRoot\scripts" -File) {
    if ($item.Name -notin $keepScripts) {
        $target=Assert-WorkspacePath $item.FullName
        Remove-Item -LiteralPath $target -Force
    }
}
foreach ($name in @('resume-remaining.cmd','docs\history')) {
    $target=Assert-WorkspacePath "$workspaceRoot\$name"
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force -Recurse }
}
$history=Assert-WorkspacePath "$workspaceRoot\reports\history_before_20261014"
New-Item -ItemType Directory -Path $history -Force | Out-Null
foreach ($item in Get-ChildItem -LiteralPath "$workspaceRoot\reports" -Force) {
    if ($item.Name -in @('history_before_20261014','transition_20261014','SELECTED_MODELS_REPORT.md','selected_results.csv','verification_selected.json')) { continue }
    $source=Assert-WorkspacePath $item.FullName
    $destination=Assert-WorkspacePath (Join-Path $history $item.Name)
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    Move-Item -LiteralPath $source -Destination $destination
}
[pscustomobject]@{ status='completed'; deleted=$deleted; removed_file_bytes=($deleted | Measure-Object -Property bytes -Sum).Sum; retained_archive=$plan.archive; history=$history } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$workspaceRoot\reports\transition_20261014\cleanup_result.json" -Encoding UTF8
Get-Content -LiteralPath "$workspaceRoot\reports\transition_20261014\cleanup_result.json" -Raw
