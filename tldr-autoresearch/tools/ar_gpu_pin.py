#!/usr/bin/env python3
"""
ar_gpu_pin.py - install / verify the GPU pin inside the autoresearch venv.

Every Python process started from <repo>/.venv (including `uv run ...` launched by the agent with an
arbitrary CUDA_VISIBLE_DEVICES) gets CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<uuid>
forced at interpreter start-up, before torch can initialise CUDA. Two independent hooks are written:

  site-packages/sitecustomize.py   imported by site.py at start-up (can be shadowed by a
                                   sitecustomize.py earlier on PYTHONPATH)
  site-packages/00_ar_gpu_pin.pth  executed by site.py while scanning site-packages; cannot be
                                   shadowed via PYTHONPATH

Neither survives `python -S`/`-I` or a re-created venv; ar_run.py calls ensure_pin() before every
run (self-healing) and records a flag when it had to repair anything.

  python3 ar_gpu_pin.py --repo $RLTLDR_ROOT/autoresearch --gpu GPU-...   # install / repair (--venv <dir> also works)
  python3 ar_gpu_pin.py --repo ... --gpu GPU-... --check                 # exit 1 if not intact
"""
import argparse
import glob
import os
import sys

SITECUSTOMIZE = '''\
# Installed by tools/ar_gpu_pin.py (RLTL;DR) - do not edit (ar_run.py restores it).
# Pins every Python process of this venv to ONE physical GPU, overriding the caller's environment.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{uuid}"
'''
# .pth files only execute lines that start with "import"; keep it a single line.
PTH = ('import os; os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", '
       'CUDA_VISIBLE_DEVICES="{uuid}")  # ar_gpu_pin.py\n')


def site_packages(venv):
    hits = sorted(glob.glob(os.path.join(venv, "lib", "python3*", "site-packages")))
    if not hits:
        raise FileNotFoundError(f"no site-packages under venv {venv} (run `uv sync` first)")
    return hits[0]


def expected_files(venv, uuid):
    sp = site_packages(venv)
    return {os.path.join(sp, "sitecustomize.py"): SITECUSTOMIZE.format(uuid=uuid),
            os.path.join(sp, "00_ar_gpu_pin.pth"): PTH.format(uuid=uuid)}


def check_pin(venv, uuid):
    """Return the list of pin files that are missing or modified (empty list == intact)."""
    bad = []
    for path, content in expected_files(venv, uuid).items():
        try:
            with open(path) as f:
                if f.read() != content:
                    bad.append(path)
        except OSError:
            bad.append(path)
    return bad


def ensure_pin(venv, uuid):
    """(Re)write any missing/modified pin file. Returns the list of files that were repaired."""
    repaired = []
    for path in check_pin(venv, uuid):
        content = expected_files(venv, uuid)[path]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, path)
        repaired.append(path)
    return repaired


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--venv", help="venv directory (e.g. <repo>/.venv)")
    g.add_argument("--repo", help="shorthand for --venv <repo>/.venv")
    ap.add_argument("--gpu", required=True, help="GPU UUID (GPU-..., see nvidia-smi -L)")
    ap.add_argument("--check", action="store_true", help="only verify; exit 1 if missing/modified")
    a = ap.parse_args()
    venv = a.venv or os.path.join(a.repo, ".venv")
    if a.check:
        bad = check_pin(venv, a.gpu)
        for p in bad:
            print(f"NOT INTACT: {p}")
        print("pin intact" if not bad else "pin broken")
        return 1 if bad else 0
    for p in ensure_pin(venv, a.gpu):
        print(f"wrote {p}")
    print("pin intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
