param(
    [string]$Report = "report/midterm/midterm.tex"
)

$ErrorActionPreference = "Stop"
$reportPath = Resolve-Path $Report
$reportDir = Split-Path $reportPath
$reportFile = Split-Path $reportPath -Leaf

Push-Location $reportDir
try {
    pdflatex -interaction=nonstopmode $reportFile
    $base = [System.IO.Path]::GetFileNameWithoutExtension($reportFile)
    if (Test-Path "$base.aux") {
        bibtex $base
        pdflatex -interaction=nonstopmode $reportFile
        pdflatex -interaction=nonstopmode $reportFile
    }
}
finally {
    Pop-Location
}
