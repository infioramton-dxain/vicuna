
$ErrorActionPreference = 'Stop'

$guardArg = '-NoProfile -WindowStyle Hidden -Command "if (-not (Get-Process tailscale-ipn -ErrorAction SilentlyContinue)) { Start-Process ''C:\Program Files\Tailscale\tailscale-ipn.exe'' }"'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $guardArg

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
$trigLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigEvening = New-ScheduledTaskTrigger -Daily -At '19:38'
Register-ScheduledTask -TaskName 'VicunaTailscaleGuard' -Action $action -Trigger @($trigLogon, $trigEvening) -Settings $settings -Description 'Vicuna: ensure the Tailscale GUI client is running (backend stays NoState without it); at logon + 19:38 before the evening push.' -Force | Out-Null
Write-Output 'OK: VicunaTailscaleGuard registered (at logon + daily 19:38)'

if (-not (Get-Process tailscale-ipn -ErrorAction SilentlyContinue)) {
    Start-Process 'C:\Program Files\Tailscale\tailscale-ipn.exe'
    Write-Output 'OK: tailscale-ipn started now'
} else {
    Write-Output 'OK: tailscale-ipn already running'
}
