#!/bin/bash
# Run ONE experiment: submits the current train.py to the experiment runner (dedicated GPU, outside this sandbox).
#   ./run.sh "<short description of the change>"
# Prints `val_bpb: X` / `peak_vram_mb: Y`, or `status: <error>` + the end of the training log. Takes ~6 minutes.
exec /usr/bin/python3 -I "@RLTLDR_ROOT@/tools/run_client.py" "$@"
