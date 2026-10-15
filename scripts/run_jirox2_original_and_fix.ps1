param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [Parameter(Mandatory = $true)][string]$GroupsConfig,
    [Parameter(Mandatory = $true)][string]$WorkspaceRoot,
    [Parameter(Mandatory = $true)][string]$FixWorkspaceRoot,
    [Parameter(Mandatory = $true)][string]$OriginalLog,
    [Parameter(Mandatory = $true)][string]$FixLog,
    [Parameter(Mandatory = $true)][string]$XdayDir
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
New-Item -ItemType Directory -Force $OutputRoot, $WorkspaceRoot, $FixWorkspaceRoot | Out-Null

$common = @(
    'scripts\reconstruct_pdl_variants.py',
    '--source-root', $SourceRoot,
    '--output-root', $OutputRoot,
    '--groups-config', $GroupsConfig,
    '--continue-on-error',
    '--overwrite',
    '--xday-dir', $XdayDir
)

$originalArgs = $common + @(
    '--top-dir', 'JIROX-II',
    '--index', 'JIROX-II/reconstruction-index.json',
    '--workspace-root', $WorkspaceRoot
)

$fixArgs = $common + @(
    '--top-dir', 'JIROX-IIFix',
    '--index', 'JIROX-IIFix/reconstruction-index.json',
    '--workspace-root', $FixWorkspaceRoot,
    '--fix-projection-u-offset'
)

"[$(Get-Date -Format s)] original batch start" | Set-Content -LiteralPath $OriginalLog
& python @originalArgs 2>&1 | Tee-Object -FilePath $OriginalLog -Append
$originalExit = $LASTEXITCODE
"[$(Get-Date -Format s)] original batch exit=$originalExit" | Add-Content -LiteralPath $OriginalLog

"[$(Get-Date -Format s)] Fix batch start" | Set-Content -LiteralPath $FixLog
& python @fixArgs 2>&1 | Tee-Object -FilePath $FixLog -Append
$fixExit = $LASTEXITCODE
"[$(Get-Date -Format s)] Fix batch exit=$fixExit" | Add-Content -LiteralPath $FixLog

if ($originalExit -ne 0 -or $fixExit -ne 0) {
    exit 1
}
exit 0
