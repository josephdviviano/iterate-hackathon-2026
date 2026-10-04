#!/bin/bash
# Run ONE experiment: submits the current train.py to the experiment runner (dedicated GPU, outside this sandbox),
# blocks until the run is done (~6 minutes) and prints the summary (see program.md).
#   ./run.sh "<short description of the change>" > run.log 2>&1
exec /usr/bin/python3 -I "@RLTLDR_ROOT@/tools/run_client.py" "$@"
