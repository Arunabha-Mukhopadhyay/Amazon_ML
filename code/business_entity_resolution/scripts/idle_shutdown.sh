#!/usr/bin/env bash
# Power the VM off after ER_IDLE_MINUTES (default 30) minutes without work.
# Run by cron every 5 minutes (installed by setup_vm.sh). "Work" means a
# pipeline process is running or the 1-minute load average is above 0.5.
# An instance shut down from inside the OS is *stopped* (not terminated), so
# files are kept and compute billing stops.
STATE=/var/tmp/er_idle_since
LIMIT=${ER_IDLE_MINUTES:-30}

busy=0
load=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($load > 0.5)}"; then busy=1; fi
if pgrep -f "er\.run|er\.prepare|er\.sample|validate_submission|pip install" >/dev/null; then busy=1; fi

now=$(date +%s)
if [ "$busy" = 1 ]; then
  rm -f "$STATE"
  exit 0
fi
[ -f "$STATE" ] || echo "$now" > "$STATE"
since=$(cat "$STATE")
if [ $(( (now - since) / 60 )) -ge "$LIMIT" ]; then
  rm -f "$STATE"
  /sbin/shutdown -h now "idle for ${LIMIT} minutes"
fi
