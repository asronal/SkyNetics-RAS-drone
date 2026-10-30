#!/bin/bash
# usage: tools/cpu_run.sh LABEL SECONDS — runs main.py (gpu) and reports total CPU %, AnyDesk CPU and the app's [PERF] lines
cd "$(dirname "$0")/.."; S=output; D=$2
DISPLAY=:0 XAUTHORITY=~/.Xauthority timeout -s INT -k 20 $((D+12)) ~/venv/bin/python main.py --display gpu > $S/run_$1.log 2>&1 &
sleep 12; read -r _ a b c d e f g h _ < /proc/stat; t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e))
ad1=$(ps -C anydesk -o times= | awk '{s+=$1} END{print s+0}')
sleep $D; read -r _ a b c d e f g h _ < /proc/stat; t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e))
ad2=$(ps -C anydesk -o times= | awk '{s+=$1} END{print s+0}')
wait
echo "$1: total CPU $(( 100 * ( (t2-t1) - (i2-i1) ) / (t2-t1) ))% of 4 cores | anydesk $(( (ad2-ad1) * 100 / D ))% of a core"
grep "\[PERF\]" $S/run_$1.log | tail -3 | sed 's/.*\[PERF\]//' | cut -c1-90
grep -c -i "sensor read failed\|THERMAL.*fail" $S/run_$1.log | sed 's/^/thermal errors logged: /'
