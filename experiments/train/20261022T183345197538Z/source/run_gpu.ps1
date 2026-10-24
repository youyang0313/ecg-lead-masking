param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ExperimentArguments)
$ErrorActionPreference = 'Stop'
$projectGpuPythonRoot = 'D:\py\Anaconda3\envs\fl_pytorch'
$projectOriginalPath = $env:Path
try {
    $env:Path = "$projectGpuPythonRoot\Library\bin;$projectGpuPythonRoot\Lib\site-packages\torch\lib;$projectGpuPythonRoot;$projectOriginalPath"
    & "$projectGpuPythonRoot\python.exe" -u "$PSScriptRoot\experiment.py" @ExperimentArguments
    if ($LASTEXITCODE -ne 0) { throw "ECG experiment exited with code $LASTEXITCODE" }
} finally {
    $env:Path = $projectOriginalPath
}
