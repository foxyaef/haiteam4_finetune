param(
    [string]$Python = '',
    [ValidateSet('xpu','cpu','cu128')][string]$Backend = 'xpu'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $Candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )
    $Python = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Python) { throw 'Python 3.12 or 3.13 not found. Pass -Python with its full path.' }
}
function Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Exe (exit $LASTEXITCODE)" }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    Checked $Python @('-m','venv','.venv')
}
$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Checked $Py @('-m','pip','install','torch==2.8.0','torchvision==0.23.0','--index-url',"https://download.pytorch.org/whl/$Backend")
Checked $Py @('-m','pip','install','-r','requirements.txt')
$Commit = '29320b6fd828f8e0987a71426cf2d961b09dfed7'
if (-not (Test-Path -LiteralPath 'vendor\RT-DETR\.git')) {
    New-Item -ItemType Directory -Force 'vendor' | Out-Null
    Checked 'git' @('clone','https://github.com/lyuwenyu/RT-DETR.git','vendor/RT-DETR')
    Checked 'git' @('-C','vendor/RT-DETR','checkout','--detach',$Commit)
}
$Actual = & git -C vendor/RT-DETR rev-parse HEAD
if ($Actual -ne $Commit) { throw 'Upstream repository commit differs; keep existing files and resolve it manually.' }
New-Item -ItemType Directory -Force 'weights','reports' | Out-Null
$Weight = 'weights/rtdetr_r50vd_6x_coco_from_paddle.pth'
if (-not (Test-Path -LiteralPath $Weight)) {
    Invoke-WebRequest -Uri 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth' -OutFile "$Weight.part"
    Move-Item -LiteralPath "$Weight.part" -Destination $Weight
}
$Hash = (Get-FileHash -LiteralPath $Weight -Algorithm SHA256).Hash.ToLower()
$Expected = (Get-Content -LiteralPath 'upstream-lock.json' -Raw | ConvertFrom-Json).checkpoint_sha256
if ($Hash -ne $Expected) { throw 'Official checkpoint SHA256 differs from project lock.' }
@{sha256=$Hash} | ConvertTo-Json | Set-Content -LiteralPath "$Weight.sha256.json" -Encoding utf8
& $Py -m pip freeze | Set-Content -LiteralPath 'reports/installed-packages.txt' -Encoding utf8
Checked $Py @('-m','pip','check')
Write-Host 'Setup complete. Next: .\.venv\Scripts\python.exe .\prepare_data.py; then .\coco.cmd check.'
