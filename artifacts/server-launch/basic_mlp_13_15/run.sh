#!/bin/sh
cd /Users/server/experiments/spectrome-basic-mlp-12-20261015 || exit 1
batch=artifacts/server-launch/basic_mlp_13_15
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=4
/usr/bin/caffeinate -i .venv/bin/python -u "$batch/launch.py" > "$batch/launcher.log" 2>&1
run_status=$?
printf '%s\n' "$run_status" > "$batch/exit_code.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$batch/finished_at.txt"
exit "$run_status"
