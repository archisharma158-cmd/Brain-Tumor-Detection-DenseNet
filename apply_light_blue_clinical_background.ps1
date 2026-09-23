$ErrorActionPreference = "Stop"

$path = Join-Path (Get-Location) "app\app.py"

if (-not (Test-Path $path)) {
    throw "app\app.py was not found. Run this script from the Brain-Tumor-Detection-DenseNet project root."
}

$original = [System.IO.File]::ReadAllText($path)

$replacement = @'
    .stApp {
        background-color: #EEF8FF;
        background-image:
            /* Soft clinical cyan illumination */
            radial-gradient(ellipse 980px 760px at 88% 4%, rgba(56, 189, 248, 0.115) 0%, rgba(125, 211, 252, 0.035) 45%, transparent 72%),
            /* Medical blue depth */
            radial-gradient(ellipse 980px 760px at 5% 94%, rgba(37, 99, 235, 0.075) 0%, rgba(59, 130, 246, 0.018) 48%, transparent 72%),
            /* Very subtle brand-pink clinical accent */
            radial-gradient(ellipse 760px 560px at 96% 72%, rgba(236, 72, 153, 0.035) 0%, transparent 64%),
            /* MRI / neural clinical contour field */
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='840' height='520' viewBox='0 0 840 520' fill='none'%3E%3Cpath d='M0 130 C130 95 230 160 360 132 C505 101 630 145 840 112' stroke='%230E7490' stroke-width='1.1' stroke-opacity='0.055' fill='none'/%3E%3Cpath d='M0 235 C150 190 275 285 430 236 C590 185 680 255 840 218' stroke='%232563EB' stroke-width='1' stroke-opacity='0.045' fill='none'/%3E%3Cpath d='M0 360 C175 315 300 405 470 355 C610 315 735 380 840 350' stroke='%230F4C81' stroke-width='1' stroke-opacity='0.04' fill='none'/%3E%3Cpath d='M40 455 C210 420 335 488 520 445 C650 415 740 456 840 438' stroke='%23EC4899' stroke-width='0.9' stroke-opacity='0.025' fill='none'/%3E%3Cline x1='150' y1='95' x2='275' y2='285' stroke='%230E7490' stroke-width='0.7' stroke-opacity='0.03'/%3E%3Cline x1='360' y1='132' x2='430' y2='236' stroke='%232563EB' stroke-width='0.7' stroke-opacity='0.03'/%3E%3Cline x1='430' y1='236' x2='470' y2='355' stroke='%230E7490' stroke-width='0.7' stroke-opacity='0.03'/%3E%3Ccircle cx='150' cy='95' r='2.5' fill='%230E7490' fill-opacity='0.06'/%3E%3Ccircle cx='275' cy='285' r='2.4' fill='%232563EB' fill-opacity='0.055'/%3E%3Ccircle cx='430' cy='236' r='3' fill='%230E7490' fill-opacity='0.055'/%3E%3Ccircle cx='470' cy='355' r='2.4' fill='%23EC4899' fill-opacity='0.035'/%3E%3C/svg%3E"),
            /* Fine radiology registration dots */
            radial-gradient(circle, rgba(14, 116, 144, 0.055) 1px, transparent 1px),
            /* Diagnostic imaging grid */
            linear-gradient(to right, rgba(37, 99, 235, 0.025) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(37, 99, 235, 0.025) 1px, transparent 1px),
            /* Light-blue clinical canvas */
            linear-gradient(180deg, #F7FCFF 0%, #EEF8FF 42%, #E6F3FC 100%);
        background-size:
            auto,
            auto,
            auto,
            840px 520px,
            34px 34px,
            72px 72px,
            72px 72px,
            100% 100%;
        background-position:
            center top,
            center top,
            center top,
            0 0,
            0 0,
            0 0,
            0 0,
            0 0;
        background-repeat:
            no-repeat,
            no-repeat,
            no-repeat,
            repeat,
            repeat,
            repeat,
            repeat,
            no-repeat;
        background-attachment: fixed;
        color: var(--ink);
    }

'@

$pattern = '(?s)    \.stApp \{.*?(?=    \[data-testid="stAppViewContainer"\] \{)'
$matches = [regex]::Matches($original, $pattern)

if ($matches.Count -ne 1) {
    throw "Expected exactly one .stApp background block, but found $($matches.Count). No file was changed."
}

$backup = "$path.before-light-blue-background.bak"
[System.IO.File]::Copy($path, $backup, $true)

$updated = [regex]::Replace($original, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacement }, 1)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($path, $updated, $utf8NoBom)

Write-Host "Applied light-blue clinical medical background to app\app.py"
Write-Host "Backup created at: $backup"
Write-Host "No buttons, content, navigation, ML, prediction, evaluation, or Grad-CAM logic was modified."

python -m py_compile app/app.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "Python syntax check passed."
}
