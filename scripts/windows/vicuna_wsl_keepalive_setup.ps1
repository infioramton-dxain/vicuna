
$ErrorActionPreference = 'Stop'

$keepAliveArg = '-NoProfile -WindowStyle Hidden -Command "Start-Process -WindowStyle Hidden wsl.exe -ArgumentList ''-d Ubuntu --exec bash -c \"flock -n /tmp/.vicuna_keepalive.lock sleep infinity\"''"'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $keepAliveArg

$settingsLogon = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
$trigLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
Register-ScheduledTask -TaskName 'VicunaWSLKeepAlive' -Action $action -Trigger $trigLogon -Settings $settingsLogon -Description 'Vicuna: hidden WSL client so the VM (and cron) survives closing the last terminal.' -Force | Out-Null
Write-Output 'OK: VicunaWSLKeepAlive registered (at logon)'

$settingsWake = New-ScheduledTaskSettingsSet -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
$trigWake = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '19:40'
Register-ScheduledTask -TaskName 'VicunaEveningWake' -Action $action -Trigger $trigWake -Settings $settingsWake -Description 'Vicuna: wake the machine at 19:40 ET weekdays so the 19:50-22:15 cron chain runs; re-arms the WSL keep-alive.' -Force | Out-Null
Write-Output 'OK: VicunaEveningWake registered (19:40 Mon-Fri, WakeToRun)'

Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d Ubuntu --exec bash -c "flock -n /tmp/.vicuna_keepalive.lock sleep infinity"'
Write-Output 'OK: keep-alive client started now'
