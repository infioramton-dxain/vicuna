#!/bin/bash

LOG_DIR="$HOME/system_logs"
mkdir -p "$LOG_DIR"
TS=$(date '+%Y-%m-%d_%H-%M-%S')
REPORT="$LOG_DIR/snapshot_${TS}.txt"

{
echo "=== SNAPSHOT: $TS ==="
echo

echo "--- CPU ---"
echo "Load average (1/5/15 min):"
cat /proc/loadavg
echo
echo "Top 5 CPU-hungry processes right now:"
ps -eo pid,%cpu,%mem,comm --sort=-%cpu | head -n 6
echo

echo "--- GPU ---"
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
else
  echo "nvidia-smi not found in this environment."
fi
echo

echo "--- Disk / SSD I/O ---"
if command -v iostat &> /dev/null; then
  iostat -x 1 2 | tail -n +1
else
  echo "(iostat not installed — install with 'sudo apt install sysstat' for richer disk stats)"
  echo "Falling back to raw /proc/diskstats read/write counters:"
  cat /proc/diskstats
fi
echo

echo "--- Memory & Swap (WSL2 view — misleading in isolation, see host below) ---"
free -h
echo
echo "Swap activity (si=swap in, so=swap out — nonzero means active thrashing):"
vmstat 1 2
echo

export PATH="$PATH:/mnt/c/Windows/System32/WindowsPowerShell/v1.0:/mnt/c/Windows/System32"
echo "--- Memory (Windows HOST view — authoritative) ---"
if command -v powershell.exe &> /dev/null; then
  timeout 30 powershell.exe -NoProfile -Command "\$os = Get-CimInstance Win32_OperatingSystem; \$totalMB = [math]::Round(\$os.TotalVisibleMemorySize/1024,0); \$freeMB = [math]::Round(\$os.FreePhysicalMemory/1024,0); \$usedMB = \$totalMB - \$freeMB; \$pct = [math]::Round(100 * \$usedMB / \$totalMB, 1); Write-Host \"Physical: \$usedMB / \$totalMB MB used (\$pct%); free = \$freeMB MB\"; \$mem = Get-CimInstance Win32_OperatingSystem; Write-Host \"Committed (virtual): \$([math]::Round((\$mem.TotalVirtualMemorySize - \$mem.FreeVirtualMemory)/1024,0)) / \$([math]::Round(\$mem.TotalVirtualMemorySize/1024,0)) MB\"" 2>&1
  rc=$?
  if [ "$rc" -eq 124 ]; then
    echo "(powershell.exe timed out after 30s — WSL interop hung; host memory reading unavailable)"
  elif [ "$rc" -ne 0 ]; then
    echo "(powershell.exe exited rc=$rc — host memory reading unavailable)"
  fi
else
  echo "(powershell.exe not on PATH — not running under WSL?)"
fi
echo

echo "=== END SNAPSHOT ==="
} | tee "$REPORT"

echo
echo "Snapshot saved to: $REPORT"
